import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


@dataclass
class Paper:
    # Unique paper ID, formatted as `{venue}_{year}_{pid}`
    pid: str
    # Year of publication
    year: int
    # Publication venue
    venue: str
    # Title of the paper
    title: Optional[str] = None
    # Authors of the paper
    authors: Optional[List[str]] = field(default_factory=list)
    # Abstract of the paper
    abstract: Optional[str] = None
    # URL to the PDF of the paper that can be directly downloaded
    pdf_url: Optional[str] = None
    # URL to the paper page on the venue website
    url: Optional[str] = None


class BaseCollection(ABC):
    def __init__(self, year: int, venue: str):
        self.year: int = year
        self.venue: str = venue
        self.papers: List[Paper] = []
        self.data_dir: str = None

    @abstractmethod
    def collect(self):
        """
        Abstract method that must be implemented by subclasses.
        This method should collect papers from the specific venue.
        """
        pass

    def add_paper(
        self,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        pdf_url: Optional[str] = None,
        url: Optional[str] = None,
    ):
        self.papers.append(
            Paper(
                pid=f"{self.venue}_{self.year}_{len(self.papers)}",
                year=self.year,
                venue=self.venue,
                title=title,
                authors=authors,
                abstract=abstract,
                pdf_url=pdf_url,
                url=url,
            )
        )

    def save_metadata(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        metadata_path = os.path.join(self.data_dir, "metadata.tsv")
        metadata_df = pd.DataFrame(self.papers)
        metadata_df.to_csv(metadata_path, sep="\t", index=False)

    def download_pdfs(self):
        pdf_dir = os.path.join(self.data_dir, "pdf")
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        for paper in tqdm(self.papers, desc="Downloading PDFs"):
            if paper.pdf_url:
                pdf_path = os.path.join(pdf_dir, f"{paper.pid}.pdf")
                if not os.path.exists(pdf_path):
                    response = requests.get(paper.pdf_url)
                    with open(pdf_path, "wb") as f:
                        f.write(response.content)
