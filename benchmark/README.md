## How to generate the test set.

1. prepare two files at these paths: `massw/raw_data/extracted_0525_0531.jsonl` and `massw/raw_data/oag_publication_selected_0525.json`
2. run `python filter_data.py` to filter out entries with no N/A
3. run `python stratified_sampling.py`
4. the test set is located at `massw/benchmark/test_set_processing/test_set.jsonl`, consisting of 1200 papers

## How to run the benchmark

run `python test.py`

a sample evalution ressults on 30 test cases is stored in `evaluation_results_30_samples.json`