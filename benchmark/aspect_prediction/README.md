# Benchmark

## Reproduce Results in the Paper

To reproduce the benchmark results across different models and prompt types, run

```bash
python benchmark/aspect_prediction/task.py --model <model> --prompt <prompt_style> --num_samples 1020
```

where:

- `<model>` is chosen from `gpt-35-turbo`, `gpt-4`, `mixtral-8x7b`.
- `<prompt_style>` is chosen from `zero-shot`, `few-shot`, `chain-of-thought`, `few-shot-cot`.

> We provide the benchmark output through a Dropbox link
> [here](https://www.dropbox.com/scl/fi/nap87vh9s2mc7v3daql5u/results_v1.zip?rlkey=m1n5vck90quwhqygiq1otn2zp&dl=0).
> You could download the results and unzip them to the
> `benchmark/aspect_prediction/outputs` directory through:
>
> ```bash
> wget "https://www.dropbox.com/scl/fi/nap87vh9s2mc7v3daql5u/results_v1.zip?rlkey=m1n5vck90quwhqygiq1otn2zp&dl=1" -O results_v1.zip
> unzip results_v1.zip -d benchmark/aspect_prediction
> rm results_v1.zip
> mv benchmark/aspect_prediction/results benchmark/aspect_prediction/outputs
> ```

After running the tasks, evaluate the outcomes by running:

```bash
python benchmark/aspect_prediction/eval.py --model_output_dir benchmark/aspect_prediction/outputs/gpt-35-turbo_zero-shot
```

---



## Adding a Custom Model to MASSW/API

To extend the functionality of MASSW by adding custom API scripts for additional models, follow these guidelines. This will allow your model to integrate seamlessly with the existing framework used for aspect prediction and evaluation.

#### 1. **Location for API Scripts**

Place your custom API scripts in the `massw/api` directory. This should be similar in structure and design to the existing scripts:

- `massw/api/api_gpt.py`
- `massw/api/api_mixtral.py`

#### 2. **Required Functions**

Each API script must include two essential functions:

- **`prompts_to_raw_output_<model_name>`**: This function processes prompts and generates raw outputs.

```python
def prompts_to_raw_output_<model_name>(messages: List[Tuple[str, str]], **other_arguments) -> pd.DataFrame:
    """
    Process prompts to generate raw outputs.

    Parameters:
    - messages (List[Tuple[str, str]], str]]): A list of tuples containing paper IDs and messages.
      'pid' is the paper ID, and 'message' is the text of the conversation or prompt.

    Returns:
    - pd.DataFrame: A DataFrame containing the processed outputs with paper IDs.
    """
    pass
```

- **`raw_output_to_dict_<model_name>`**: This function parses raw outputs into a dictionary format.

  ```python
  def raw_output_to_dict_<model_name>(output_path: str) -> Dict[str, str]:
      """
      Convert raw outputs into a dictionary mapping from paper ID to output.

      Parameters:
      - output_path (str): The file path to the output directory where the results are stored.

      Returns:
      - Dict[str, str]: A dictionary mapping each paper ID to its corresponding output.
      """
      pass
  ```

#### 3. **Modify the Task Processing Function**

Update the `process_task` function in `benchmark/aspect_prediction/task.py` to handle your custom model by calling your new API functions. Additionally, adapt the `postprocess_output` function in `benchmark/aspect_observer/eval.py` to support the evaluation of your model's outputs.