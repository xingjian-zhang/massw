"""
This script facilitates the automated benchmarking of research prompts.

The script supports multiple models and prompt types
and is designed to work with large sets
of test data asynchronously.
"""

import argparse
import os
import sys

import jsonlines as jl
from prompts import (SYSTEM_PROMPT, future_work_recommendation,
                     idea_generation, method_recommendation,
                     outcome_prediction, predict_title)
from utils import (MODEL_CHOICES, PROMPT_CHOICES, allow_self_signed_https,
                   load_examples, save_results)

from massw.models import gpt_azure, mixtral_azure

sys.path.append("../..")

allow_self_signed_https(True)

few_shot_examples, cot_examples = load_examples()


def prepare_messages(model, task_name, prompt_type, main_prompt):
    """Prepare the messages based on the task and prompt type."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if model == "mixtral-8x7b":
        format_instruction = "\nNever use double quotes in your output. \
                             Use single quotes instead.\n"
        messages = [
            {"role": "user", "content": SYSTEM_PROMPT + format_instruction},
            {"role": "assistant", "content": "I got it. \
             Please give me further instructions!"}
        ]

    if prompt_type in ["few-shot", "few-shot-cot"]:
        examples = few_shot_examples if prompt_type == "few-shot"\
            else cot_examples
        for example in examples.get(task_name, []):
            messages.extend([
                {"role": "user", "content": example["user"]},
                {"role": "assistant", "content": example["assistant"]}
            ])

    if prompt_type == "chain-of-thought":
        main_prompt += "Let's think step by step. \
                        You should first present you reasoning. \
                        After that, the final prediction should start after \
                        the marker 'Prediction:'."

    messages.append({"role": "user", "content": main_prompt})

    return messages


def process_task(generate_prompt_fn, test_cases, task_name, **kwargs):
    """Process the tasks and retrieve chat completions."""
    messages = []
    for entry in test_cases:
        main_prompt, _ = generate_prompt_fn(entry)
        message = prepare_messages(kwargs['model'],
                                   task_name,
                                   kwargs['prompt_type'],
                                   main_prompt)
        messages.append((entry['pid'], message))

    model = kwargs['model']
    if model == "mixtral-8x7b":
        chat_results = mixtral_azure.prompts_to_raw_output(messages)
    elif model in ["gpt-35-turbo", "gpt-4"]:
        chat_results = gpt_azure.prompts_to_raw_output(messages,
                                                 model,
                                                 kwargs.get('tpm'))
    else:
        raise ValueError(f"Model {model} not supported. \
                         You can modify the code here \
                         to support custom models.")

    return chat_results


def main():
    """Execute main function to process tasks."""
    parser = argparse.ArgumentParser(description="Process benchmarking \
                                     of academic paper prompts.")
    parser.add_argument("--test_data",
                        type=str,
                        default="data/benchmark_0531.jsonl")
    parser.add_argument("--output_dir",
                        type=str,
                        default=False)
    parser.add_argument("--model",
                        type=str,
                        default="gpt-35-turbo")
    parser.add_argument("--prompt",
                        type=str,
                        default="zero-shot")
    parser.add_argument("--num_samples",
                        type=int,
                        default=5)
    args = parser.parse_args()

    if args.model not in MODEL_CHOICES:
        raise ValueError(f"Model {args.model} not supported. \
                         Choose from {MODEL_CHOICES}")

    if args.prompt not in PROMPT_CHOICES:
        raise ValueError(f"Prompt type {args.prompt} not supported. \
                         Choose from {PROMPT_CHOICES}")

    if not args.output_dir:
        args.output_dir = os.path.join("benchmark",
                                       "aspect_prediction",
                                       "outputs",
                                       f"{args.model}_{args.prompt}")

    # Load test data
    with jl.open(args.test_data) as file:
        test_data = [record for record, _ in
                     zip(file, range(args.num_samples))]

    tasks = [
        ("idea_generation", idea_generation),
        ("method_recommendation", method_recommendation),
        ("outcome_prediction", outcome_prediction),
        ("future_work_recommendation", future_work_recommendation),
        ("title_prediction", predict_title)
    ]

    tokens_per_minute = {"gpt-35-turbo": 40000,
                         "gpt-4": 10000,
                         "mixtral-8x7b": None}

    for task_name, generate_prompt_fn in tasks:
        print(f"Processing task: {task_name}")
        chat_results = process_task(
            generate_prompt_fn,
            test_data,
            task_name,
            model=args.model,
            prompt_type=args.prompt,
            tpm=tokens_per_minute[args.model]
        )
        print(f"{chat_results = }")
        save_results(chat_results, args.output_dir, task_name)


if __name__ == "__main__":
    main()
