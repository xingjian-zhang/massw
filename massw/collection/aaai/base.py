"""Base classes for paper collection from academic conferences."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
import requests
from tqdm import tqdm


@dataclass
class Paper:
    """
    Represents a research paper with its metadata.

    Attributes:
        pid: Unique paper ID, formatted as `{venue}_{year}_{pid}`
        year: Year of publication
        venue: Publication venue
        title: Title of the paper
        authors: Authors of the paper
        abstract: Abstract of the paper
        pdf_url: URL to the PDF of the paper that can be directly downloaded
        url: URL to the paper page on the venue website
    """

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
    """
    Abstract base class for collecting papers from academic conferences.

    This class provides common functionality for collecting, storing, and
    downloading papers from various academic venues.

    Attributes:
        year: The year of the conference
        venue: The venue code (e.g., 'neurips', 'acl')
        papers: List of collected papers
        data_dir: Directory to store the collected data
    """

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
        raise NotImplementedError("Subclasses must implement this method.")

    def add_paper(
        self,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        abstract: Optional[str] = None,
        pdf_url: Optional[str] = None,
        url: Optional[str] = None,
    ):
        """
        Add a paper to the collection.

        Args:
            title: Title of the paper
            authors: List of authors
            abstract: Abstract of the paper
            pdf_url: URL to the PDF file
            url: URL to the paper page on the venue website
        """
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
        """
        Save the metadata of collected papers to a TSV file.

        The metadata is saved to {data_dir}/metadata.tsv.
        """
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        metadata_path = os.path.join(self.data_dir, "metadata.tsv")
        metadata_df = pd.DataFrame(self.papers)
        metadata_df.to_csv(metadata_path, sep="\t", index=False)

    def download_pdfs(self):
        """
        Download PDF files for all papers in the collection.

        The PDFs are saved to {data_dir}/pdf/{pid}.pdf.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
            }
        pdf_dir = os.path.join(self.data_dir, "pdf")
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        if  os.path.exists(os.path.join(self.data_dir, "metadata.tsv")):
            path = os.path.join(self.data_dir, "metadata.tsv")
            df = pd.read_csv(path, sep='\t')
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
                pdf_url = row['pdf_url']
                pid = row['pid']
                pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
                response = requests.get(pdf_url, headers=headers,timeout=30)
                with open(pdf_path, "wb") as f:
                        f.write(response.content)

