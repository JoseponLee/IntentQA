import pandas as pd
import json



if __name__ == '__main__':
    spilts = ['train', 'val', 'test']
    vocab = {}
    cnt = 0
    for spilt in spilts:
        csv_path = '/mnt/d/program/NExT-OE-baseline/dataset/nextqa/' + spilt + '.csv'
        data = pd.read_csv(csv_path)
        all_answers = list(data['answer'])
        for ans in all_answers:
            if ans not in vocab.keys():
                vocab[ans] = cnt
                cnt += 1
    vocab = json.dumps(vocab)
    f = open('vocab.json', 'w')
    f.write(vocab)
    f.close()