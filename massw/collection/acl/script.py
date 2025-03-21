import os
import time
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

from massw.collection import BaseCollection


class ACLCollection(BaseCollection):
    def __init__(self, year: int, base_url: str = "https://aclanthology.org"):
        super().__init__(year, "acl")
        self.data_dir = os.path.join(os.path.dirname(__file__), f"data_{year}")
        self.base_url = base_url

    def collect(self):
        url = f"https://aclanthology.org/events/acl-{self.year}/"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the page: {response.status_code}")
            return None
        soup = BeautifulSoup(response.content, "html.parser")
        paper_links = soup.select("span.d-block > strong > a.align-middle")
        for i, link in enumerate(tqdm(paper_links, desc="Processing Papers")):
            time.sleep(1)
            rel_url = link.get("href")
            paper_url = urljoin(self.base_url, rel_url)
            pdf_url = f"{paper_url[:-1]}.pdf"
            title = link.get_text().strip()
            if (
                ("long" in rel_url or "short" in rel_url)
                and ("acl-long.0" not in rel_url)
                and ("acl-short.0" not in rel_url)
            ):
                paper_response = requests.get(paper_url)
                if paper_response.status_code == 200:
                    paper_soup = BeautifulSoup(paper_response.content, "html.parser")
                    bibtex_element = paper_soup.select_one("pre#citeBibtexContent")
                    bibtex_text = bibtex_element.get_text().strip()
                    author_match = re.search(
                        r'author\s*=\s*"\s*(.*?)\s*"', bibtex_text, re.DOTALL
                    )
                    if author_match:
                        raw_authors = author_match.group(1)
                        author_list = [
                            author.replace(",", "").strip()
                            for author in raw_authors.replace("\n", " ").split(" and ")
                        ]
                        authors = ", ".join(author_list)
                    else:
                        authors = None
                    abstract_match = re.search(
                        r'abstract\s*=\s*"\s*(.*?)\s*"', bibtex_text, re.DOTALL
                    )
                    abstract = (
                        abstract_match.group(1).strip() if abstract_match else None
                    )
                    self.add_paper(
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        pdf_url=pdf_url,
                        url=paper_url,
                    )
        self.save_metadata()

    def download_pdfs(self):
        """
        Download PDF files for all papers in the collection.

        The PDFs are saved to {data_dir}/pdf/{pid}.pdf.
        """
        pdf_dir = os.path.join(self.data_dir, "pdf")
        if not os.path.exists(pdf_dir):
            os.makedirs(pdf_dir)
        if os.path.exists(os.path.join(self.data_dir, "metadata.tsv")):
            path = os.path.join(self.data_dir, "metadata.tsv")
            df = pd.read_csv(path, sep="\t")
            for index, row in tqdm(
                df.iterrows(), total=len(df), desc="Downloading PDFs"
            ):
                pdf_url = row["pdf_url"]
                pid = row["pid"]
                pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
                response = requests.get(pdf_url, timeout=30)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)


if __name__ == "__main__":
    collection = ACLCollection(year=2024)
    if not os.path.exists(os.path.join(collection.data_dir, "metadata.tsv")):
        collection.collect()
    #collection.download_pdfs()
