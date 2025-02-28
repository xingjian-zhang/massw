import os
from joblib import Memory

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from massw.collection import BaseCollection

memory = Memory(os.path.join(os.path.dirname(__file__), "cache"), verbose=0)


@memory.cache
def get_paper_pdf_url(base_url: str, year: int) -> str:
    """Get the PDF URL for a NeurIPS paper from its virtual site URL.

    Args:
        base_url (str): The URL of the paper's virtual site page on neurips.cc
        year (int): The conference year of the paper

    Returns:
        str: The direct URL to the paper's PDF file, or None if the PDF link cannot be found

    Example:
        >>> url = get_paper_pdf_url("https://neurips.cc/virtual/2024/poster/72031", 2024)
        >>> print(url)
        'https://proceedings.neurips.cc/paper_files/paper/2024/file/abc123-Paper-Conference.pdf'
    """
    # Request the HTML content of the base URL
    response = requests.get(base_url, timeout=10)
    if response.status_code != 200:
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the PDF link - typically a link with 'PDF' or 'Paper' text
    pdf_link = soup.find("a", {"class": "btn", "title": "Paper"})

    if pdf_link and "href" in pdf_link.attrs:
        # Extract the hash from the HTML link
        html_url = pdf_link["href"]
        # Convert HTML URL to PDF URL format
        if "hash" in html_url:
            # Extract the hash part and construct the PDF URL
            hash_part = html_url.split("/")[-1].split("-")[0]
            return f"https://proceedings.neurips.cc/paper_files/paper/{year}/file/{hash_part}-Paper-Conference.pdf"

    # If we couldn't find or parse the PDF link, return None
    return None


class NeurIPSCollection(BaseCollection):
    def __init__(self, year: int):
        super().__init__(year, "neurips")
        self.data_dir = os.path.join(os.path.dirname(__file__), f"data_{year}")

    def collect(self):
        raw_path = os.path.join(self.data_dir, "raw.tsv")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw data file not found at {raw_path}. Please see the README for instructions."
            )

        raw_df = pd.read_csv(raw_path, sep="\t")
        raw_df = raw_df[raw_df["type"] == "Poster"]
        bar = tqdm(raw_df.iterrows(), total=len(raw_df))
        success_count = 0
        for _, row in bar:
            pdf_url = get_paper_pdf_url(row["virtualsite_url"], self.year)
            self.add_paper(
                title=row["name"],
                authors=row["speakers/authors"].split(", "),
                abstract=row["abstract"],
                pdf_url=pdf_url,
            )
            if pdf_url:
                success_count += 1
            bar.set_description(
                f"Collected {success_count}/{len(raw_df)} paper pdf urls."
            )
        self.save_metadata()


if __name__ == "__main__":
    collection = NeurIPSCollection(year=2024)
    if not os.path.exists(os.path.join(collection.data_dir, "metadata.tsv")):
        collection.collect()
    collection.download_pdfs()
