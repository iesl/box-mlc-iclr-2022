from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from allennlp.data import Vocabulary
from .hierarchy_reader import HierarchyReader, HierarchyInfo
import networkx as nx
import logging
from fractions import Fraction

logger = logging.getLogger(__name__)


@HierarchyReader.register("cooccurrences-csv")
class CooccurrencesCSVReader(HierarchyReader):
    """
    Reads a csv file consisting of cooccurrances.

    It assumes that:

        1. the row1 and col1 are the label names

        2. Cij (i!=j) is a Fraction #(yi & yj)/#yj

        3. Cii is #yi/#total_examples
    """

    def __init__(
        self,
        vocab: Vocabulary,
        filepath: Path,
        delimiter: Optional[str] = None,
    ):
        """
        Args:
            vocab: Model's vocab. Not passed in config but is created internally.
            filepath: path to the edgelist file
            delimiter: Delimiter used in the edgelist file
        """
        self.delimeter = delimiter
        # super init has to be called last
        super().__init__(vocab, filepath)

    def read(self) -> HierarchyInfo:
        """
        Read the csv file with a square metrix with entries of type `Fraction`.

        It does the following:
            1. Reads the csv file with first row as column header and first column as index, both which are assumed to contain label names
            2. Checks whether the Cooccurrences are supplied for all the labels seen in the data. If some are missing, those are ignored using mask.
            3. Also masks the diagonal entries (these correspond to marginals).

        Returns:
            Instance of `HierarchyInfo`
        """
        df = pd.read_csv(self.filepath, index_col=0, sep=self.delimeter, low_memory=False)
        assert len(df) == len(
            df.columns
        ), f"cooccurences csv should be square but is {(len(df), len(df.columns))}"
        cooccurrences_available = set(df.columns)
        all_labels_in_data = set(self.get_all_labels())

        if all_labels_in_data < cooccurrences_available:
            logger.info(
                f"Cooccurrence counts available for more labels "
                f"({len(cooccurrences_available)}) that in the data "
                f"({len(all_labels_in_data)})"
            )
        missing_cooccurrences = []

        if cooccurrences_available < all_labels_in_data:
            # we will create the missing columns and rows
            # but will mask the values in training
            missing_cooccurrences = list(
                (all_labels_in_data - cooccurrences_available)
            )
            logger.warning(
                f"Cooccurrences missing: {'|'.join(missing_cooccurrences)}"
            )

            for col in missing_cooccurrences:
                # add missing row and column with 1 as constant value
                # value does not matter as it will be masked
                df[col] = 1
                df.loc[col] = 1

        sorted_labels = sorted(all_labels_in_data, key=self.get_label_index)
        df_ordered = df[sorted_labels].reindex(sorted_labels)
        df_ordered = df_ordered.applymap(lambda x: float(Fraction(x)))
        df_mask = pd.DataFrame(
            True, index=df_ordered.index, columns=df_ordered.columns
        )

        for label in missing_cooccurrences:
            # mask the missing cooccurrences
            df_mask[label] = False
            df_mask.loc[label] = False

        adjacency_matrix = torch.tensor(df_ordered.values, requires_grad=False)
        mask: torch.BoolTensor = torch.tensor(
            df_mask.values, dtype=torch.bool, requires_grad=False
        )  # type: ignore
        assert mask.shape == adjacency_matrix.shape
        # mask the diagonal elements
        mask.fill_diagonal_(False)

        return {"adjacency_matrix": adjacency_matrix, "mask": mask}
