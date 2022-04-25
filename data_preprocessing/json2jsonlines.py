from typing import List, Tuple, Union, Dict, Any, Optional
import json
import sys
import argparse
try:
    from .utils import isnotebook
except ImportError:
    from utils import isnotebook
import logging
sys.path.append('../')
from tqdm import tqdm
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a datafile from .json to .jsonl"
    )
    parser.add_argument("input_file", metavar="input_file",type=Path, help="Input filepath")
    parser.add_argument("-o", "--output-file", type=Path, help="Output filepath. "
                        "If no specified same filepath as the input will be used but with .jsonl extension")
    if isnotebook():
        import shlex  # noqa

        args_str = (
            " ../.data/blurb_genre_collection/sample_train.json "
        )
        args = parser.parse_args(shlex.split(args_str))
    else:
        args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    with open(args.input_file) as f:
        logger.info(f'Reading {args.input_file}')
        input_data: Dict = json.load(f)
    if args.output_file is None:
        args.output_file = args.input_file.with_suffix('.jsonl')
    with open(args.output_file, 'w') as f:
        logger.info(f'Writing to {args.output_file}')
        for line in tqdm(input_data):
            f.write(json.dumps(line))
            f.write('\n')


if isnotebook() or (__name__ == "__main__"):
    args = get_args()
    main(args)


