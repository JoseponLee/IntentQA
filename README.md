# ICCV2023 - IntentQA: Context-aware Video Intent Reasoning

## **Introduction**

The project is described in our paper [IntentQA: Context-aware Video Intent Reasoning](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_IntentQA_Context-aware_Video_Intent_Reasoning_ICCV_2023_paper.pdf) (ICCV2023, Oral).

Among the recent flourishing studies on cross-modal vision-language understanding, video question answering (VideoQA) is one of the most prominent to support interactive AI with the ability to understand and communicate dynamic visual scenarios via natural languages. Despite its popularity, VideoQA is still quite challenging, because it demands the models to comprehensively understand the videos to correctly answer questions, which include not only factual but also inferential ones. The former directly asks about the visual facts (e.g., humans, objects, actions, etc.), while the latter (inference VideoQA) requires logical reasoning of latent variables (e.g., the spatial, temporal and causal relationships among entities, mental states, etc.) beyond observed visual facts . The future trend for AI is to study inference VideoQA beyond factoid VideoQA , requiring more reasoning ability beyond mere recognition. In this paper, we propose a new task called IntentQA, i.e., a special kind of inference VideoQA that focuses on intent reasoning. 

![img](https://0x0007e3.feishu.cn/space/api/box/stream/download/asynccode/?code=NTI4MGIyYWJjYmFkMmQzODBiNjU2ZGQ2Y2M1NmYxOTdfV2lSWk96WVRtZzdjeDRVaW1FTzFjYXpvNmtUTHJrWU1fVG9rZW46TUZvcWJidnB4b2N3YXF4TG1CS2NnQ3lsbk9iXzE3MDI5NzczMzE6MTcwMjk4MDkzMV9WNA)

## **Dataset**

Please download the **pre-computed features** and **original videos** from [here](https://www.alipan.com/s/diEEWQc5rPq) (password:56sb),

There are 3 folders:

- `Videos`: This directory contains all the original videos of the dataset, named with `video_id`. All videos are in MP4 format.
- `region_feat_n`: This folder contains pre-computed bounding box features.
- `frame_feat`: This folder includes pre-computed frame features.

Please download the **QA annotations** from [here](https://drive.google.com/drive/folders/1dtds2e3ddHQ5YyauwC3d1SiSe_K5F-Xa?usp=drive_link). There are 3 files (```train.csv```,```val.csv```,```test.csv``` ):

In each annotation file, the initial columns follow the same format as in `NExT-QA`. Building upon the `NExT-QA` foundation, we have introduced additional annotations, adding extra columns to the dataset. 

*  `action`, `lemma`, and `lemma_id`: Specifically, we have annotated `action`, `lemma`, and `lemma_id`. These columns highlight actions in the current QA that trigger intentions, either self or others', along with the lemmatized forms of these actions and their corresponding IDs after categorizing them into synonymous groups.

* `id`, `pos_id`, and `neg_id`: Furthermore, in the `train.csv` file, we have also added `id`, `pos_id`, and `neg_id` annotations. The `id` column denotes the row number of the data, while the `pos_id` and `neg_id` columns indicate the row numbers (`id`) of data in the train set that form positive and negative cases, respectively, in relation to the current row's data.

## **Results**

| Model            | Text Rep. | CW    | CH    | TP&TN | Total                                                        | Result File                                                  |
| ---------------- | --------- | ----- | ----- | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| EVQA             | GloVe     | 25.92 | 34.54 | 25.52 | 27.27                                                        |                                                              |
| CoMem            | GloVe     | 30.00 | 28.69 | 28.95 | 29.52                                                        |                                                              |
| HGA              | GloVe     | 32.00 | 30.64 | 31.05 | 31.54                                                        |                                                              |
| HME              | GloVe     | 34.40 | 34.26 | 29.14 | 33.08                                                        |                                                              |
| HQGA             | GloVe     | 33.20 | 34.26 | 36.57 | 34.21                                                        |                                                              |
| CoMem            | BERT      | 47.68 | 54.87 | 39.05 | 46.77                                                        |                                                              |
| HGA              | BERT      | 44.88 | 50.97 | 39.62 | 44.61                                                        |                                                              |
| HME              | BERT      | 46.08 | 54.32 | 40.76 | 46.16                                                        |                                                              |
| HQGA             | BERT      | 48.24 | 54.32 | 41.71 | 47.66                                                        |                                                              |
| VGT              | BERT      | 51.44 | 55.99 | 47.62 | 51.27                                                        |                                                              |
| **Blind GPT**    | BERT      | 52.16 | 61.28 | 43.43 | **51.55**                                                    | [Here](https://drive.google.com/file/d/161zkUQsyUKvHuFp2qPCk5Gk5vtc28flT/view?usp=drive_link) |
| **Ours w/o GPT** | BERT      | 55.28 | 61.56 | 47.81 | **[54.50](https://drive.google.com/file/d/1C2clniRU44UqxDi_9R5ZMqpneHNM_j6T/view?usp=drive_link)** | [Here](https://drive.google.com/file/d/1C2clniRU44UqxDi_9R5ZMqpneHNM_j6T/view?usp=drive_link) |
| **Ours**         | BERT      | 58.40 | 65.46 | 50.48 | **57.64**                                                    | [Here](https://drive.google.com/file/d/17x66SNgj9bWit6LpgDEY8hMpR68gSP6k/view?usp=drive_link) |
| **Human**        | -         | 77.76 | 80.22 | 79.05 | **78.49**                                                    | [Here](https://drive.google.com/file/d/1VAdgeV3WGGlPLUhLzX9vdba_Ni7MxJH3/view?usp=drive_link) |

## **Demo**

Here is a [demo](https://vimeo.com/896083218?share=copy) that briefly summarizes our work.

## **Citation**

```JSON
@InProceedings{Li_2023_ICCV,
    author    = {Li, Jiapeng and Wei, Ping and Han, Wenjuan and Fan, Lifeng},
    title     = {IntentQA: Context-aware Video Intent Reasoning},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {11963-11974}
}
```
