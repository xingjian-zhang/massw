# CVPR Collection

1. Metadata Crawling:
 The script extracts raw metadata from the CVPR website `https://openaccess.thecvf.com/CVPR{year}?day=all` and saves the data in TSV format. It contains about 2716 papers

3. Collecting PDF URLs:
Run the script to extract and collect the URLs of the paper PDFs. This process takes approximately 1 hour and 30 minutes.

4.  Downloading PDFs:
After collecting the URLs, run the download_pdfs() function to fetch all the PDFs. This step is estimated to take around 10 minutes.

# Statistic
Our script retrieves metadata for a total of 2716 papers from the CVPR 2024 conference.
