"""Evaluate experiment results based on the model generated output (file)."""
import argparse
import json
import sys

import nest_asyncio
import pandas as pd
from utils import TASK2GT, TASK_NAMES, postprocess_cot

from massw.metrics import compute_metrics, flatten_metrics
from massw.models import gpt_azure, mixtral_azure

sys.path.append("..")
nest_asyncio.apply()


def postprocess_output(model_output_dir,
                       reference_path,
                       used_cot=False,
                       model_type="gpt"):
    """
    Process model output files to match predictions with references.

    Args:
        model_output_dir (str): Directory containing the output files.
        reference_path (str): Path to the file containing reference data.
        used_cot (bool): Flag to determine if COT processing is needed.
        model_type (str): Type of model used to adjust processing logic.

    Returns:
        dict: A dictionary containing predictions and references by task.
    """
    results = {}
    with open(reference_path, "r", encoding="utf-8") as f:
        references = [json.loads(line) for line in f]
    id2ref = {r["pid"]: r for r in references}

    for task_name in TASK_NAMES:
        gt_name = TASK2GT[task_name]
        model_path = f"{model_output_dir}/{task_name}.tsv"

        if model_type == "gpt":
            id2predictions = gpt_azure.raw_output_to_dict(model_path)
        elif model_type == "mixtral":
            id2predictions = mixtral_azure.raw_output_to_dict(model_path)
        else:
            raise ValueError(f"Model type {model_type} not supported.")

        if used_cot:
            for pid in id2predictions:
                id2predictions[pid] = postprocess_cot(id2predictions[pid])

        results[task_name] = {
            "predictions": list(id2predictions.values()),
            "references": [id2ref[pid][gt_name] for pid in id2ref.keys()
                           if pid in id2predictions]
        }

    return results


def main():
    """Run main function to process and compute evaluation metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output_dir",
        type=str,
        help="Path to the model output dir.",
        default="benchmark/aspect_prediction/outputs/gpt-35-turbo_zero-shot",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="Path to the references file.",
        default="data/benchmark_0531.jsonl",
    )
    parser.add_argument(
        "--used_cot",
        action="store_true",
        help="Used COT.",
    )
    args = parser.parse_args()
    model_type = "gpt" if "gpt" in args.model_output_dir else "mixtral"

    results = postprocess_output(
        args.model_output_dir,
        args.reference_path,
        args.used_cot,
        model_type=model_type,
    )
    metrics_output_path = f"{args.model_output_dir}/metrics.tsv"

    metrics = {}

    for task_name, task_results in results.items():
        print(f"Processing task: {task_name}")
        predictions = task_results["predictions"]
        references = task_results["references"]
        metrics[task_name] = flatten_metrics(
            compute_metrics(
                predictions,
                references,
                metric_names=[
                    "bleu", "rouge", "cosine", "bertscore", "bleurt"
                ],
            ))
        print(f"Processed task: {task_name}")
        print(metrics[task_name])

    df = pd.DataFrame(metrics)
    df.to_csv(metrics_output_path, index=True, sep="\t")


if __name__ == "__main__":
    main()
