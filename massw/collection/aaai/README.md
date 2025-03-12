# AAAI Collection

1. Metadata Crawling:
The script crawls raw metadata from the AAAI archive website `https://ojs.aaai.org/index.php/AAAI/issue/archive` and saves the data in TSV format. This archive contains all the AAAI conference papers. For example, the archive currently lists 21 technical tracks in 2024. If the conference adds more papers or if a year’s papers are split across multiple pages, the script will adapt by checking URLs formatted like `https://ojs.aaai.org/index.php/AAAI/issue/archive/{index}` and gathering all relevant pages.

2. Collecting PDF URLs:
Run the script to extract and collect the URLs of the paper PDFs. This process takes approximately 1 hour and 20 minutes.

3.  Downloading PDFs:
After collecting the URLs, run the download_pdfs() function to fetch all the PDFs. This step is estimated to take around 1 hour and 30 minutes.

# Statistic
Our script retrieves metadata for 2,865 papers from the AAAI-2024 conference. This number aligns closely with external sources, some of which report figures like 2,867 papers, and it confirms that the conference accepted over 2,500 papers overall. This comprehensive metadata collection underscores the scale and breadth of contributions presented at AAAI-2024.


