import os
from joblib import Memory
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import base
from base import BaseCollection
import time
import re
import os
__file__ = os.path.abspath("script.py")


def get_aaai_track_urls(archive_url):

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
    soup = BeautifulSoup(response.text, 'html.parser')
    
    track_info = []
    issue_summaries = soup.select('.obj_issue_summary')
    
    print(f"Found {len(issue_summaries)} issue summaries")
    
    for summary in issue_summaries:
        title_element = summary.select_one('a.title')
        if title_element:
            title = title_element.text.strip()
            url = title_element['href']
            if not title.startswith('AAAI-23'):
               track_info.append(url)
    
    print(f"Extracted {len(track_info)} track URLs.")
    return track_info




class AAAICollection(BaseCollection):
    def __init__(self, year: int, base_url: str):
        super().__init__(year, "aaai")
        self.data_dir = os.path.join(os.path.dirname(__file__), f"data_{year}")
        self.base_url = base_url
    
    def collect(self):
        track_urls = get_aaai_track_urls(self.base_url)
    
        if not track_urls:
            print("No track URLs found. Checking if base_url itself contains papers...")
            track_urls = [self.base_url]  # Fallback to using base_url directly
        for track_index, track_url in enumerate(tqdm(track_urls, desc="Processing tracks", position=0, leave=True)):
            headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/112.0.0.0 Safari/537.36"
            }
            response = requests.get(track_url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch the track page. Status code: {response.status_code}")
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            paper_links = []
            for article_div in soup.select('.obj_article_summary'):
                title_element = article_div.select_one('.title')
                if title_element and title_element.a:
                    paper_title = title_element.a.text.strip()
                    paper_url = title_element.a['href']
                    paper_links.append((paper_title, paper_url))       
            print(f"Found {len(paper_links)} papers in track {track_index + 1}. Starting to fetch details...")
            for i, (title, url) in enumerate(paper_links):
                time.sleep(1)
                try:
                    paper_response = requests.get(url, headers=headers)
                    if paper_response.status_code == 200:
                        paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
            
                        authors_div = paper_soup.select_one('.authors')
                        authors = []
                        if authors_div:
                            for author in authors_div.select('.name'):
                                authors.append(author.text.strip())
                        abstract_div = paper_soup.select_one('.item.abstract')
                        abstract_text = ""
                        if abstract_div:
                            label = abstract_div.select_one('.label')
                            if label:
                                label.decompose()
                            abstract_text = abstract_div.get_text(strip=True)
                        pdf_url = ""
                        pdf_link = paper_soup.select_one('a.obj_galley_link.pdf')
                        if pdf_link and 'href' in pdf_link.attrs:
                            pdf_url = pdf_link['href']
                        self.add_paper(
                            title=title,
                            authors=authors,
                            abstract=abstract_text,
                            pdf_url=pdf_url,
                            url=url
                        )
                    else:
                        print(f"Failed to fetch paper at {url}. Status code: {paper_response.status_code}")
                    
                except Exception as e:
                    print(f"Error processing paper {url}: {str(e)}")
                
            print(f"Completed track {track_index + 1}/{len(track_urls)}")
        self.save_metadata()



if __name__ == "__main__":
    collection = AAAICollection(year=2024,base_url="https://ojs.aaai.org/index.php/AAAI/issue/archive")
    if not os.path.exists(os.path.join(collection.data_dir, "metadata.tsv")):
        collection.collect()
    collection.download_pdfs()