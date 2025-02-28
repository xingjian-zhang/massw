# Conference Paper Collection

This folder contains the script and instructions for collecting the conference paper metadata and full text.

## Supported conferences
- ✅ - Supported
- ❌ - Unsupported
- 🚧 - Work in Progress

| Conference | Metadata | PDF | Parsed Full Text |
|:----------:|:--------:|:---:|:----------------:|
| AAAI       | 🚧      | 🚧 | 🚧              |
| ACL        | 🚧      | 🚧 | 🚧              |
| CHI        | 🚧      | 🚧 | 🚧              |
| CVPR       | 🚧      | 🚧 | 🚧              |
| ECCV       | 🚧      | 🚧 | 🚧              |
| EMNLP      | 🚧      | 🚧 | 🚧              |
| ICCV       | 🚧      | 🚧 | 🚧              |
| ICLR       | 🚧      | 🚧 | 🚧              |
| ICML       | 🚧      | 🚧 | 🚧              |
| IJCAI      | 🚧      | 🚧 | 🚧              |
| KDD        | 🚧      | 🚧 | 🚧              |
| NAACL      | 🚧      | 🚧 | 🚧              |
| NeurIPS    | 🚧      | 🚧 | 🚧              |
| SIGIR      | 🚧      | 🚧 | 🚧              |
| SIGMOD     | 🚧      | 🚧 | 🚧              |
| VLDB       | 🚧      | 🚧 | 🚧              |
| WWW        | 🚧      | 🚧 | 🚧              |

## Folder Structure for each conference

Each conference has its own folder. In the folder, there are 
- a `README.md` file for instructions
- a `script.py` file for scraping the metadata and full text
- multiple `data_<year>` folders for storing the metadata and full text
    - `metadata.tsv`: Metadata for the conference papers.
    - `pdf`: Folder for storing the PDF files.
        - `<pid>.pdf`: PDF file for the conference paper.
    - `parsed_full_text`: Folder for storing the parsed full text.
        - `<pid>.txt`: Parsed full text for the conference paper.

```
massw
├── collection
    ├── aaai
        ├── README.md
        ├── script.py
        ├── data_2024
            ├── metadata.tsv
            ├── pdf
            └── parsed_full_text
        ├── data_2025
            ├── metadata.tsv
            ├── pdf
            └── parsed_full_text
```
