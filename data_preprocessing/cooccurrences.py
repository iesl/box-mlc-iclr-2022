#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Script to compute label-label cooccurances"""
import argparse

try:
    from .utils import isnotebook
except ImportError:
    from utils import isnotebook
from pathlib import Path
import json
import sys

sys.path.append("../")
from box_mlc.dataset_readers.common import JSONTransform
from box_mlc.dataset_readers.utils import smart_read
from allennlp.common.params import Params
import itertools
import numpy as np
import pandas as pd
from fractions import Fraction
import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# In[2]:


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cooccurance score for a dataset"
    )
    parser.add_argument("-i", "--input-file", type=Path)
    parser.add_argument(
        "-o", "--output-file", type=Path, default=Path("cooccurrences.csv")
    )
    parser.add_argument(
        "-l",
        "--label-field-name",
        default="labels",
        help="name of the field in (transformed) json that contains labels as list",
    )
    parser.add_argument(
        "-t",
        "--json-transform",
        type=(lambda x: JSONTransform.from_params(Params({"type": x}))),
        default=JSONTransform.from_params(Params(dict(type="identity"))),
        help='Registered child of "dataset_readers.common.JSONTransform"',
    )
    parser.add_argument(
        "-n",
        "--num-labels",
        type=int,
        default=None,
        help="No. of labels in the dataset",
    )

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


# In[3]:


if __name__ == "__main__":
    args = get_args()
    label_sets = [
        args.json_transform(ex)[args.label_field_name]
        for ex in smart_read(args.input_file, num_labels=args.num_labels)
    ]
    num_examples = len(label_sets)
    all_labels = set([l for s in label_sets for l in s])
    all_pairs = list(itertools.product(all_labels, repeat=2))
    label_df = pd.DataFrame(
        Fraction(0.0), columns=all_labels, index=all_labels
    )
    logger.info("counting co-occurances")

    for label_set in tqdm(label_sets):
        for a, b in itertools.product(label_set, repeat=2):
            label_df[b][a] += 1
    logger.info("get pair-wise conditional probabilities")

    for a, b in tqdm(all_pairs):
        if a != b:
            # indexing in df is weird
            # [a][b] accesses col=a, row=b
            label_df[b][a] /= label_df[b][b]

    for l in tqdm(all_labels):
        label_df[l][l] /= num_examples
    #breakpoint()
    logger.info(f"Writing to {args.output_file}")
    label_df.to_csv(args.output_file, index_label="labels")


# In[ ]:
