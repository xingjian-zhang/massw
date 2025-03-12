import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from massw.collection import BaseCollection


def get_aaai_track_urls(archive_url,year:int):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/112.0.0.0 Safari/537.36"
    }

    print(f"Fetching archive page: {archive_url}")
    response = requests.get(archive_url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch the archive page. Status code: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")

    track_info = []
    issue_summaries = soup.select(".obj_issue_summary")

    print(f"Found {len(issue_summaries)} issue summaries")
    conference_tag = f"AAAI-{str(year)[-2:]}" 

    for summary in issue_summaries:
        title_element = summary.select_one("a.title")
        if title_element:
            title = title_element.text.strip()
            url = title_element["href"]
            if conference_tag in title:
                track_info.append(url)

    print(f"Extracted {len(track_info)} track URLs.")
    return track_info


class AAAICollection(BaseCollection):
    def __init__(self, year: int, base_url: str = "https://ojs.aaai.org/index.php/AAAI/issue/archive",):  ## all 2024 technique tracks for AAAI conference is in this url, maybe be updated if AAAI release more papers.
        super().__init__(year, "aaai")
        self.data_dir = os.path.join(os.path.dirname(__file__), f"data_{year}")
        self.base_url = base_url

    def collect(self):
        track_urls = get_aaai_track_urls(self.base_url,self.year)

        if not track_urls:
            print("No track URLs found. Checking if base_url itself contains papers...")
            track_urls = [self.base_url]
        for track_index, track_url in enumerate(tqdm(track_urls, desc="Processing tracks", position=0, leave=True)):
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/112.0.0.0 Safari/537.36"
            }
            response = requests.get(track_url, headers=headers)
            if response.status_code != 200:
                print(
                    f"Failed to fetch the track page. Status code: {response.status_code}"
                )
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            paper_links = []
            for article_div in soup.select(".obj_article_summary"):
                title_element = article_div.select_one(".title")
                if title_element and title_element.a:
                    paper_title = title_element.a.text.strip()
                    paper_url = title_element.a["href"]
                    paper_links.append((paper_title, paper_url))
            print(
                f"Found {len(paper_links)} papers in track {track_index + 1}. Starting to fetch details..."
            )
            for i, (title, url) in enumerate(paper_links):
                time.sleep(1)
                try:
                    paper_response = requests.get(url, headers=headers)
                    if paper_response.status_code == 200:
                        paper_soup = BeautifulSoup(paper_response.text, "html.parser")

                        authors_div = paper_soup.select_one(".authors")
                        authors = []
                        if authors_div:
                            for author in authors_div.select(".name"):
                                authors.append(author.text.strip())
                        abstract_div = paper_soup.select_one(".item.abstract")
                        abstract_text = ""
                        if abstract_div:
                            label = abstract_div.select_one(".label")
                            if label:
                                label.decompose()
                            abstract_text = abstract_div.get_text(strip=True)
                        pdf_url = ""
                        pdf_link = paper_soup.select_one("a.obj_galley_link.pdf")
                        if pdf_link and "href" in pdf_link.attrs:
                            pdf_url = pdf_link["href"]
                        self.add_paper(
                            title=title,
                            authors=authors,
                            abstract=abstract_text,
                            pdf_url=pdf_url,
                            url=url,
                        )
                    else:
                        print(
                            f"Failed to fetch paper at {url}. Status code: {paper_response.status_code}"
                        )

                except Exception as e:
                    print(f"Error processing paper {url}: {str(e)}")

            print(f"Completed track {track_index + 1}/{len(track_urls)}")
        self.save_metadata()

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
        if os.path.exists(os.path.join(self.data_dir, "metadata.tsv")):
            path = os.path.join(self.data_dir, "metadata.tsv")
            df = pd.read_csv(path, sep="\t")
            for index, row in tqdm(
                df.iterrows(), total=len(df), desc="Downloading PDFs"
            ):
                pdf_url = row["pdf_url"]
                pid = row["pid"]
                pdf_path = os.path.join(pdf_dir, f"{pid}.pdf")
                response = requests.get(pdf_url, headers=headers, timeout=30)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)


if __name__ == "__main__":
    collection = AAAICollection(year=2024)
    if not os.path.exists(os.path.join(collection.data_dir, "metadata.tsv")):
        collection.collect()
    collection.download_pdfs()
