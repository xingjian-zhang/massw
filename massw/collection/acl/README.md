# ACL Collection

1. Metadata Crawling:
The script crawls raw metadata from the ACL website `https://aclanthology.org/events/acl-{year}/` and saves the data in TSV format. This archive contains all 941 papers of long paper
and short paper in ACL conference

3. Collecting PDF URLs:
Run the script to extract and collect the URLs of the paper PDFs. This process takes approximately 50 minutes.

4.  Downloading PDFs:
After collecting the URLs, run the download_pdfs() function to fetch all the PDFs. This step is estimated to take around 10 minutes.

# Statistic
Our script retrieves metadata for 941 papers from the ACL-2024 conference. This number is corresponding to the ACL official website.
