from skmultilearn.dataset import load_from_arff,save_to_arff
from wcmatch import glob
from pathlib import Path
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import coo_matrix,dok_matrix,lil_matrix
import logging
import arff
import sys
import argparse
try:
    from .utils import isnotebook
except ImportError:
    from utils import isnotebook
sys.path.append('../')

logger = logging.getLogger(__name__)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize arff 10 fold meka files' feature space"
    )
    parser.add_argument("-i", "--input-file", type=Path, help="name of the glob file path")
    parser.add_argument(
        "-n",
        "--num-labels",
        type=int,
        default=None,
        help='No. of labels in the dataset',
    )
    parser.add_argument(
        '-s',
        '--save-sparse',
        action="store_true",
        default=False,
        help="Save output in the sparse format")
    if isnotebook():
        import shlex  # noqa

        args_str = (
            "-i ../.data/blurb_genre_collection/sample_train.json -o "
            "../.data/blurb_genre_collection/sample_train_cooccurrences.csv "
            "-t from-blurb-genre"
        )
        args = parser.parse_args(shlex.split(args_str))
    else:
        args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    num_labels = args.num_labels
    file_path = args.input_file
    save_sparse = args.save_sparse
    lengths = []
    filepaths = []
    data = []
    y_data = []
    print(file_path, num_labels)
    for file_ in glob.glob(str(file_path), flags=glob.EXTGLOB):
        print(f"Reading {file_}")
        filepaths.append(file_)
        x, y, feature_names, label_names = load_from_arff(
            file_,
            label_count=num_labels,
            return_attribute_definitions=True,
        )
        lengths.append(x.toarray().shape[0])
        data += x.toarray().tolist()
        y_data+=y.toarray().tolist()

    #Normalise
    scaler = MinMaxScaler()
    data_new = scaler.fit_transform(data)

    #re-dump normalised data
    start=0
    for idx,file_ in enumerate(filepaths):
        print(file_)
        new_file_path = file_.split('.arff')[0]+'-normalised.arff'
        with open(file_, 'r') as f:
            d = arff.load(f)
            X = data_new[start:start+lengths[idx]]
            y = y_data[start:start+lengths[idx]]
            y_prefix = len(X[0])
            x_prefix = 0
            if save_sparse:
                dn = [{} for r in range(len(X))]
            else:
                dn = [[0 for r in range(len(X[0])+len(y[0]))] for p in range(len(X))]
    #         breakpoint()
            for i in range(len(X)):
                for j in range(len(X[0])):
                    try:
                        if X[i][j]!=0:
                            dn[i][x_prefix + j] = X[i][j]
                    except:
                        print(i,j)
    #         breakpoint()
            for p in range(len(y)):
                for q in range(len(y[0])):
                    if y[p][q]!=0:
                        dn[p][y_prefix + q] = y[p][q]
            d['data']=dn
            start = start+lengths[idx]
        with open(new_file_path, 'w') as f:
            arff.dump(d, f)
        
        





