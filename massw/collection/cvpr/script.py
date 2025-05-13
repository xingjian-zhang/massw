import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from massw.collection import BaseCollection
from urllib.parse import urljoin


class CVPRCollection(BaseCollection):
    def __init__(self, year: int, base_url: str = "https://openaccess.thecvf.com/"):
        super().__init__(year, "cvpr")
        self.data_dir = os.path.join(os.path.dirname(__file__), f"data_{year}")
        self.base_url = base_url
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/117.0.0.0 Safari/537.36"
            )
        }

    def collect(self):
        url = f"{self.base_url}CVPR{self.year}?day=all"
        response = requests.get(url, headers=self.headers, timeout=30)
        if response.status_code != 200:
            print(f"Failed to fetch data: Status code {response.status_code}")

        soup = BeautifulSoup(response.text, "html.parser")
        paper_titles = soup.find_all("dt", class_="ptitle")
        for i, title_element in enumerate(tqdm(paper_titles, desc="Processing Papers")):
            time.sleep(1)
            title_link = title_element.find("a")
            # get title
            title = title_link.text.strip()
            # get paper_url
            paper_url = urljoin(self.base_url, title_link["href"])
            info_element = title_element.find_next_sibling("dd")
            info_element = info_element.find_next_sibling("dd")
            pdf_link = info_element.find("a", text="pdf")
            # get pdf_url
            pdf_url = urljoin(self.base_url, pdf_link["href"]) if pdf_link else None
            # get authors
            bibtex_div = info_element.find("div", class_="bibref")
            authors = None
            if bibtex_div:
                bibtex_text = bibtex_div.text.strip()
                author_match = re.search(r"author\s*=\s*{([^}]+)}", bibtex_text)
                if author_match:
                    authors_original = author_match.group(1).strip()
                    author_parts = authors_original.split(" and ")
                    transformed_authors = []
                    for author in author_parts:
                        parts = [part.strip() for part in author.split(",")]
                        if len(parts) >= 2:
                            transformed_authors.append(f"{parts[0]} {parts[1]}")
                        else:
                            transformed_authors.append(author)

            authors = ", ".join(transformed_authors)
            # get abstract
            try:
                response = requests.get(
                    paper_url, headers=self.headers, timeout=30, verify=False
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")
                abstract_div = soup.find("div", id="abstract")
                if abstract_div:
                    abstract_text = abstract_div.text.strip()
                    abstract_text = re.sub(r"\s+", " ", abstract_text)
            except:
                print(f"Failed to fetch {paper_url}")
                continue

            self.add_paper(
                title=title,
                authors=authors,
                abstract=abstract_text,
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
                response = requests.get(pdf_url, verify=False, timeout=30)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)


if __name__ == "__main__":
    collection = CVPRCollection(year=2024)
    if not os.path.exists(os.path.join(collection.data_dir, "metadata.tsv")):
        collection.collect()
    collection.download_pdfs()
