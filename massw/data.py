"""Data loading and processing functions."""

from dataclasses import dataclass
import numpy as np

import os
import pandas as pd
import jsonlines as jl
from typing import List, Union

from massw.download import download_dataset

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")


@dataclass
class Workflow:
    """Workflow dataclass."""

    id: str  # Unique ID
    title: str  # Title of the publication
    abstract: str  # Abstract of the publication
    year: int  # Year of publication
    venue: str  # Publication venue
    context: str  # Context of the workflow
    key_idea: str  # Key idea of the workflow
    method: str  # Method used in the workflow
    outcome: str  # Outcome of the workflow
    projected_impact: str  # Projected impact of the workflow
    data: dict = None  # Additional data


class MASSWDataset:
    """MASSW dataset class."""

    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame):
        """Initialize the dataset.

        Args:
            data (pd.DataFrame): The data containing the workflows.
            metadata (pd.DataFrame): The metadata containing the workflow
                information.
        """
        self.merged_data = data.join(metadata.set_index("id"), on="id")
        self.merged_data = self.merged_data.set_index("id")
        self.merged_data = self.merged_data.fillna(np.nan).replace([np.nan],
                                                                   [None])

    def __len__(self):
        """Return the number of workflows in the dataset."""
        return len(self.merged_data)

    def _get_by_position(self, pos: int) -> Workflow:
        """Return a workflow by its position.

        Args:
            pos (int): The position of the workflow.

        Returns:
            Workflow: The workflow object.
        """
        row = self.merged_data.iloc[pos]
        return Workflow(id=self.merged_data.index[pos],
                        title=row["title"],
                        abstract=row["abstract"],
                        year=row["year"],
                        venue=row["venue"],
                        context=row["context"],
                        key_idea=row["key_idea"],
                        method=row["method"],
                        outcome=row["outcome"],
                        projected_impact=row["projected_impact"],
                        data=row["data"])

    def _get_by_unique_id(self, id_my: str) -> Workflow:
        """Return a workflow by its unique ID.

        Args:
            id_my (str): The unique ID of the workflow.

        Returns:
            Workflow: The workflow object.
        """
        row = self.merged_data.loc[id_my]
        return Workflow(id=id_my,
                        title=row["title"],
                        abstract=row["abstract"],
                        year=row["year"],
                        venue=row["venue"],
                        context=row["context"],
                        key_idea=row["key_idea"],
                        method=row["method"],
                        outcome=row["outcome"],
                        projected_impact=row["projected_impact"],
                        data=row["data"])

    def __getitem__(self, key):
        """Return a workflow by its unique ID or position."""
        if isinstance(key, int):
            return self._get_by_position(key)
        if isinstance(key, str):
            return self._get_by_unique_id(key)
        raise TypeError("Invalid key type.")

    def __iter__(self):
        """Return an iterator over the workflows."""
        for idx in range(len(self)):
            yield self[idx]

    def __repr__(self):
        """Return a string representation of the dataset."""
        return f"MASSWDataset({len(self)} workflows)"

    def search(self,
               query: str,
               return_ids=False) -> Union[List[Workflow], List[str]]:
        """Search for workflows containing a query string by title.

        Args:
            query (str): The query string to search for.
            return_ids (bool): Whether to return the IDs of the workflows.
            If true, returns a list of IDs.
            Otherwise, returns a list of Workflows.

        Returns:
            list: A list of workflows or IDs containing the query string.
        """
        mask = self.merged_data["title"].str.contains(query,
                                                      case=False,
                                                      na=False)
        ids = mask[mask].index
        if return_ids:
            return ids.tolist()
        return [self._get_by_unique_id(id) for id in ids]


def load_massw(version: str = "v1") -> MASSWDataset:
    """Load the massw dataset.

    Args:
        version (str): The version of the dataset to load.

    Returns:
        MASSWDataset: The MASSW dataset object.
    """
    data_path = os.path.join(DATA_DIR, f"massw_{version}.tsv")
    metadata_path = os.path.join(DATA_DIR, f"massw_metadata_{version}.jsonl")
    if not os.path.exists(data_path) or not os.path.exists(metadata_path):
        download_dataset(version)
    data = pd.read_csv(data_path, sep="\t")
    metadata = []
    with jl.open(metadata_path) as f:
        for line in f:
            metadata.append(line)
    metadata = pd.DataFrame(metadata)
    return MASSWDataset(data, metadata)
