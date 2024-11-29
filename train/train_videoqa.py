import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import collections
from util import compute_aggreeings, AverageMeter, get_mask, mask_tokens
import os.path as osp
import json
import time
import os
import cv2
import numpy as np
import json


def process_attention_clip_to_dict(attention_clip_arg, sample):
    video_id = sample["video_id"]
    QID = sample['question_id']
    attention_clip_arg = attention_clip_arg.cpu().numpy()
    for i in range(len(video_id)):
        video_path = '/mnt/d/program/data/images_6fps_bbox/' + str(video_id[i])
        if osp.exists(video_path):
            # print(attention_clip_arg[i], video_id[i])
            for clipid in attention_clip_arg[i]:
                start_ind = clipid * 4
                for j in range(4):
                    image_path = '/mnt/d/program/data/images_6fps_bbox/' + str(video_id[i]) + '/' + str(start_ind+j+1).zfill(3) +'.jpg'
                    output_dir = '/mnt/d/program/data/images_6fps_bbox_QA/' + str(video_id[i]) + '/' + str(QID[i])
                    output_path = '/mnt/d/program/data/images_6fps_bbox_QA/' + str(video_id[i]) + '/' + str(QID[i]) + '/'\
                                  + str(start_ind+j+1).zfill(3) +'.jpg'
                    if not osp.exists(output_dir):
                        os.makedirs(output_dir)
                    image = cv2.imread(image_path)
                    mask = np.ones_like(image)
                    # print(mask.shape)
                    mask[:, :, 0] *= 0
                    mask[:, :, 1] *= 0
                    mask[:, :, 2] *= 255

                    overlapping = cv2.addWeighted(image, 0.8, mask, 0.2, 0)
                    cv2.imwrite(output_path, overlapping)



def eval(model, data_loader, a2v, args, test=False):
    model.eval()
    count = 0
    metrics, counts = collections.defaultdict(int), collections.defaultdict(int)
    if args.GPT_result != '':
        with open(args.GPT_result) as GPT_file:
            GPT_contents = GPT_file.read()
            GPT_results = json.loads(GPT_contents)


    with torch.no_grad():
        if not args.mc:
            model.module._compute_answer_embedding(a2v)
        results = {}
        for i, batch in enumerate(data_loader):
            answer_id, answer, video_o, video_f, question, question_id = (
                batch[0]["answer_id"],
                batch[0]["answer"],
                batch[0]["video_o"].cuda(),
                batch[0]["video_f"].cuda(),
                batch[0]["question"].cuda(),
                batch[0]['question_id']
            )
            
            video_len = batch[0]["video_len"]
            seq_len = batch[0]["seq_len"]
            question_mask = (question > 0).float()
            answer_mask = (answer > 0).float()
            video_mask = get_mask(video_len, video_f.size(1)).cuda()
            count += answer_id.size(0)
            video = (video_o, video_f)
            if not args.mc:
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len
                )
                topk = torch.topk(predicts, dim=1, k=10).indices.cpu()
                if args.dataset != "ivqa":
                    answer_id_expanded = answer_id.view(-1, 1).expand_as(topk)
                else:
                    answer_id = (answer_id / 2).clamp(max=1)
                    answer_id_expanded = answer_id
                metrics = compute_aggreeings(
                    topk,
                    answer_id_expanded,
                    [1, 10],
                    ["acc", "acc10"],
                    metrics,
                    ivqa=(args.dataset == "ivqa"),
                )
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(topk.numpy()[bs,0]), 'answer':int(answer_id.numpy()[bs])}
            else:
                fusion_proj, answer_proj, _, attention_clip_arg = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer.cuda(),
                    seq_len = seq_len
                )
                # process_attention_clip_to_dict(attention_clip_arg, batch[0])
                # predicts = fusion_proj.squeeze()

                # if i == 0:
                #     save_npy = fusion_proj.cpu().numpy()
                # else:
                #     save_npy = np.vstack((save_npy, fusion_proj.cpu().numpy()))
                # print(save_npy.shape)

                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                softmax_predicts = torch.softmax(predicts, dim=1)
                gpt_results_mask = torch.zeros_like(softmax_predicts).cuda()
                if args.GPT_result != '':
                    for bs, qid in enumerate(question_id):
                        # print(softmax_predicts[bs])
                        # print('gpt: ', GPT_results[qid]['prediction'])
                        gpt_results_mask[bs][GPT_results[qid]['prediction']] = 0.85
                        # print(gpt_results_mask[bs])

                    softmax_predicts += gpt_results_mask
                    softmax_predicts /= 2
                    # print(softmax_predicts[bs])
                predicted = torch.max(softmax_predicts, dim=1).indices.cpu()
                metrics["acc"] += (predicted == answer_id).sum().item()
                for bs, qid in enumerate(question_id):
                    results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer':int(answer_id.numpy()[bs])}
                    # print(results[qid])
        # print(save_npy.shape)
        # np.save('VGT_B10_our_dataset_2.npy', save_npy)

    step = "val" if not test else "test"
    
    for k in metrics:
        # print(metrics[k], count)
        v = metrics[k] / count
        logging.info(f"{step} {k}: {v:.2%}")
        break

    return metrics["acc"] / count, results


def train(model, train_loader, a2v, optimizer, criterion, triplet_loss, align_att, scheduler, epoch, args, tokenizer):
    model.train()
    running_vqa_loss, running_acc, running_mlm_loss = (
        AverageMeter(),
        AverageMeter(),
        AverageMeter(),
    )
    for i, batch in enumerate(train_loader):
        loss = 0
        predicts_list = []
        attention_list = []
        video_id_list = []
        for index in range(len(batch)):
            video_id, answer_id, answer, video_o, video_f, question = (
                batch[index]["video_id"],
                batch[index]["answer_id"],
                batch[index]["answer"],
                batch[index]["video_o"].cuda(),
                batch[index]["video_f"].cuda(),
                batch[index]["question"].cuda(),
            )
            video_len = batch[index]["video_len"]
            question_mask = (question > 0).float()
            answer_mask = (answer>0).float()
            video_mask = (
                get_mask(video_len, video_f.size(1)).cuda() if args.max_feats > 0 else None
            )
            # mask according attention mask
            # if index == 1 and video_id == video_id_list[0]:
            #     # print(video_mask.shape, attention_list[0].shape)
            #     for i in range(attention_list[0].shape[0]):
            #         for j in range(attention_list[0].shape[1]):
            #             arg = attention_list[0][i][j]
            #             video_mask[i][arg] = 0
                # print(video_mask)

            video = (video_o, video_f)
            N = answer_id.size(0)
            seq_len = batch[index]["seq_len"]
            if not args.mc:
                # a = time.time()
                model.module._compute_answer_embedding(a2v)
                predicts = model(
                    video,
                    question,
                    text_mask=question_mask,
                    video_mask=video_mask,
                    seq_len = seq_len
                )
            else:
                # a = time.time()
                fusion_proj, answer_proj, clipwise_node, attention_clip_arg = model(
                    video,
                    question,
                    text_mask=answer_mask,
                    video_mask=video_mask,
                    answer=answer.cuda(),
                    ans_id = answer_id.cuda(),
                    seq_len = seq_len,
                )
                # predicts = fusion_proj.squeeze()
                fusion_proj = fusion_proj.unsqueeze(2)
                # print(fusion_proj.shape, answer_proj[:, :5, :].shape)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze()
                predicts_list.append(clipwise_node)
                attention_list.append(attention_clip_arg)
            video_id_list.append(video_id)

            if args.dataset == "ivqa":
                a = (answer_id / 2).clamp(max=1).cuda()
                vqa_loss = criterion(predicts, a)
                predicted = torch.max(predicts, dim=1).indices.cpu()
                predicted = F.one_hot(predicted, num_classes=len(a2v))
                running_acc.update((predicted * a.cpu()).sum().item() / N, N)
            else:
                vqa_loss = criterion(predicts, answer_id.cuda())
                predicted = torch.max(predicts, dim=1).indices.cpu()
                # unk = 0
                # for k in range(predicted.shape[0]):
                #     if predicted[k].item() == answer_id[k].item() and answer_id[k].item() == 0:
                #         unk += 1
                running_acc.update((predicted == answer_id).sum().item() / N, N)

            if args.mlm_prob:
                max_seq_len = args.qmax_words
                if args.mc > 0:
                    tmp_id = [aid+(args.mc*i) for i, aid in enumerate(answer_id)]
                    inputs = answer.view(N*args.mc, -1)[tmp_id,:]
                    question_mask = (inputs>0).float()
                    max_seq_len = args.amax_words
                else:
                    inputs = batch[0]["question"]
                inputs, labels = mask_tokens(
                    inputs, tokenizer, mlm_probability=args.mlm_prob
                )
                mlm_loss = model(
                    video,
                    question=inputs.cuda(),
                    labels=labels.cuda(),
                    text_mask=question_mask,
                    video_mask=video_mask,
                    max_seq_len=max_seq_len,
                    mode="mlm",
                )
                mlm_loss = mlm_loss.mean()
                loss_ = mlm_loss + vqa_loss
            else:
                loss_ = vqa_loss
            loss += loss_

        pos_sample = align_att(predicts_list[0], predicts_list[1])[0]
        neg_sample = align_att(predicts_list[0], predicts_list[2])[0]

        # print(neg_sample)

        loss += triplet_loss(predicts_list[0], pos_sample, neg_sample)
        optimizer.zero_grad()
        loss.backward()
        if args.clip:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip)
        optimizer.step()
        scheduler.step()

        running_vqa_loss.update(loss.detach().cpu().item(), N)
        if args.mlm_prob:
            running_mlm_loss.update(mlm_loss.detach().cpu().item(), N)


        if (i + 1) % (len(train_loader) // args.freq_display) == 0:
            if args.mlm_prob:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Training acc: {running_acc.avg:.2%}, MLM loss: {running_mlm_loss.avg:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch + 1}/{args.epochs}, Lr:{optimizer.param_groups[0]['lr']}, Progress: {float(i + 1) / len(train_loader):.4f}, VQA loss: "
                    f"{running_vqa_loss.avg:.4f}, Train acc: {running_acc.avg:.2%}"
                )
            running_acc.reset()
            running_vqa_loss.reset()
            running_mlm_loss.reset()







