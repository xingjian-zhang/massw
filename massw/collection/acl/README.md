# ACL Collection

1. Metadata Crawling:
 The script extracts raw metadata from the ACL website `https://aclanthology.org/events/acl-{year}/` and saves the data in TSV format. We only consider metadata for all 941 long and short papers presented at the ACL conference.

3. Collecting PDF URLs:
Run the script to extract and collect the URLs of the paper PDFs. This process takes approximately 50 minutes.

4.  Downloading PDFs:
After collecting the URLs, run the download_pdfs() function to fetch all the PDFs. This step is estimated to take around 10 minutes.

# Statistic
Our script retrieves metadata for a total of 941 papers from the ACL 2024 conference, encompassing both long and short papers. This figure is consistent with the official records available on the ACL website, ensuring the accuracy of the collected data.
