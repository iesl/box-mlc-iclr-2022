import numpy as np
import argparse
import json
from pathlib import Path
from sklearn.utils import shuffle
import jsonlines

def retreive_fold_name(source, fold):
    return (source.split('/')[-1]).split('.')[0] + f'-fold-{fold}.jsonl'

def retreive_data(source_file):
    with open(source_file) as f:
        data = [json.loads(line) for line in f]
    return data

def split_data_into_folds(data, num_folds):
    data = shuffle(data, random_state=24)
    length = int(len(data) / num_folds)
    folds = []
    for i in range(num_folds):
        if i+1!= num_folds:
            folds+= [data[i*length:(i+1)*length]]
        else:
            folds+= [data[i*length:]]
    return folds

def write_data_to_file(destination, data):
    with jsonlines.open(destination, 'w') as writer:
        writer.write_all(data)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process xml files')
    parser.add_argument('--source_file', type=str, required=True,
                        help='Source file path')
    parser.add_argument('--destination_folder', type=str, default='./',
                        help='destination folder path')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Num folds to split into')
    args = parser.parse_args()
    if not args.source_file.endswith('.jsonl'):
        print(f'Source {args.source_file} file is not of jsonl format')
    else:
        data = retreive_data(args.source_file)
        folds = split_data_into_folds(data, args.num_folds)
        for idx, fold_data in enumerate(folds):
            destination_path = Path(args.destination_folder)/retreive_fold_name(args.source_file, idx)
            write_data_to_file(destination_path, fold_data)
