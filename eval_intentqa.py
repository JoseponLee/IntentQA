import os.path as osp
from util import load_file
import argparse

# Mapping of question type codes to their descriptive names
map_name = {
    'CW': 'Why',
    'CH': 'How',
    'TN': 'Bef&Aft',
    'TC': 'When',
    'DC': 'Cnt',
    'DL': 'Loc',
    'DO': 'Other',
    'C': 'Acc_C',
    'T': 'Acc_T',
    'D': 'Acc_D'
}


def accuracy_metric(sample_list_file, result_file):
    # Load the sample list
    sample_list = load_file(sample_list_file)

    # Initialize groups for each question type
    group = {key: [] for key in ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']}

    for _, row in sample_list.iterrows():
        qns_id = f"{row['video_id']}_{row['qid']}"
        qtype = str(row['type'])

        # Combine temporal questions of previous and next as 'TN'
        if qtype == 'TP':
            qtype = 'TN'

        # Append the question ID to the appropriate group
        if qtype in group:
            group[qtype].append(qns_id)

    # Load predictions
    preds = load_file(result_file)

    # Initialize accuracy and count dictionaries
    group_acc = {key: 0 for key in ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']}
    group_cnt = {key: 0 for key in ['CW', 'CH', 'TN', 'TC', 'DC', 'DL', 'DO']}
    overall_acc = {'C': 0, 'T': 0, 'D': 0}
    overall_cnt = {'C': 0, 'T': 0, 'D': 0}
    all_acc = 0
    all_cnt = 0

    # Calculate accuracy for each group
    for qtype, qns_ids in group.items():
        cnt = 0
        acc = 0
        for qid in qns_ids:
            cnt += 1
            answer = preds[qid]['answer']
            pred = preds[qid]['prediction']
            if answer == pred:
                acc += 1

        group_cnt[qtype] = cnt
        group_acc[qtype] += acc
        overall_acc[qtype[0]] += acc
        overall_cnt[qtype[0]] += cnt
        all_acc += acc
        all_cnt += cnt

    # Update overall accuracy and counts
    for qtype, value in overall_acc.items():
        group_acc[qtype] = value
        group_cnt[qtype] = overall_cnt[qtype]

    # Filter out qtypes with zero counts
    filtered_qtypes = [qtype for qtype in group_acc if group_cnt[qtype] > 0]

    # Print the header for valid qtypes
    for qtype in filtered_qtypes:
        print(map_name[qtype], end='\t')
    print('')

    # Print the accuracy for valid qtypes
    for qtype in filtered_qtypes:
        acc = group_acc[qtype]
        cnt = group_cnt[qtype]
        accuracy = (acc * 100.0 / cnt) if cnt > 0 else 0.00
        print('{:.2f}'.format(accuracy), end='\t')
    print('')

    # Print overall accuracy
    overall_accuracy = (all_acc * 100.0 / all_cnt) if all_cnt > 0 else 0.00
    print('Acc: {:.2f}'.format(overall_accuracy))


def main(result_file, mode='val'):
    dataset_dir = '../data/datasets/intentqa/'
    sample_list_file = osp.join(dataset_dir, f"{mode}.csv")
    print(f'Evaluating {result_file}')
    accuracy_metric(sample_list_file, result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate IntentQA Model Predictions")
    parser.add_argument("--mode", type=str, default='val', choices=['val', 'test'],
                        help="Mode to evaluate: 'val' or 'test'")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder where the result files are stored")

    args = parser.parse_args()
    res_dir = osp.join('../data/save_models/intentqa/', args.folder)
    mode = args.mode
    model_prefix = 'res'
    result_file = osp.join(res_dir, f"{mode}-{model_prefix}.json")

    main(result_file, mode)