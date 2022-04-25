from typing import List, Tuple, Union, Dict, Any, Optional
from allennlp.data.instance import Instance
from pathlib import Path
from box_mlc.dataset_readers.arff_reader import ARFFReader
import json
import jsonlines


def print_instances(instances: List[Instance], how_many: int = 5) -> None:
    for i, instance in enumerate(instances):
        print(instance)

        if i >= how_many - 1:
            break


def smart_read(path: Path, **kwargs) -> List[Dict]:
    data = []
    if path.suffix == ".arff":
        if kwargs['num_labels'] is None:
            raise ValueError(
                "No. of labels is needed"
                f" to read an .arff file but is None"
            )
        reader = ARFFReader(num_labels=kwargs['num_labels'])
        data = list(reader.read_internal(str(path.absolute())))
    else:
        with open(path) as f:
            if path.suffix == ".json":
                data = json.load(f)
            elif path.suffix == ".jsonl":
                data = list(jsonlines.Reader(f))
            else:
                raise ValueError(
                    f"file extension can only be .json/.jsonl/.arff but is {path.suffix}"
                )

    return data
