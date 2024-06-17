"""This script includes utility functions for benchmarking scripts."""
import ssl
import json
import os

MODEL_CHOICES = ["gpt-35-turbo", "gpt-4", "mixtral-8x7b"]
PROMPT_CHOICES = ["zero-shot", "few-shot", "chain-of-thought", "few-shot-cot"]


def allow_self_signed_https(allowed):
    """Config SSL settings to allow self-signed certificates."""
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(
            ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def load_examples():
    """Load few-shot and chain-of-thought examples from files."""
    with open("data/few_shot_examples.json", "r", encoding="utf-8") as f:
        few_shot_examples = json.load(f)
    with open("data/cot_examples.json", "r", encoding="utf-8") as f:
        cot_examples = json.load(f)
    return few_shot_examples, cot_examples


def save_results(chat_results, output_dir, task_name):
    """
    Save the chat results to a TSV file.

    Args:
        chat_results (DataFrame): DataFrame containing the chat results.
        output_dir (str): Directory path to save the result files.
        task_name (str): Name of task which will be used to name the output.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Serializing dictionary entries in 'output' column, if present
    if 'output' in chat_results.columns:
        chat_results['output'] = chat_results['output'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else x
        )

    output_path = os.path.join(output_dir, f"{task_name}.tsv")
    chat_results.to_csv(output_path, sep="\t", index=False)


TASK_NAMES = [
    "idea_generation",
    "method_recommendation",
    "outcome_prediction",
    "future_work_recommendation",
    "title_prediction",
]

TASK2GT = {
    "idea_generation": "key_idea",
    "method_recommendation": "method",
    "outcome_prediction": "outcome",
    "future_work_recommendation": "future_impact",
    "title_prediction": "title",
}


def postprocess_cot(output: str):
    """
    Extract the actual prediction from the output string.

    Args:
        output (str): The output string containing the prediction.

    Returns:
        str: The extracted prediction or the original output
        if no marker is found.
    """
    marker_index = output.find("Prediction:")
    if marker_index != -1:
        actual_prediction = output[marker_index + len("Prediction:"):].strip()
        return actual_prediction

    return output
