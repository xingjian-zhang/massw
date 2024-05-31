import sys
sys.path.append('..')

import asyncio
import json
from massw.api_gpt import Batch
from test_prompts import (
    load_test_set,
    idea_generation,
    method_recommendation,
    outcome_prediction,
    future_work_recommendation,
    predict_title,
    system_prompt,
)
from massw.metrics import compute_all_metrics

# Define a function to process each task separately
async def process_task(task_name, generate_prompt_fn, test_cases, model="gpt-3.5-turbo"):
    # Initialize batch processing
    batch = Batch(tpm=10000)
    
    # Add chat completion requests for the task prompts
    for entry in test_cases:
        prompt = generate_prompt_fn(entry)
        await batch.add(
            "chat.completions.create",
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
    
    # Run the batch processing for chat completions
    chat_results = await batch.run()
    
    # Extract the outputs
    outputs = chat_results["result"].apply(lambda x: x["choices"][0]["message"]["content"]).tolist()
    
    return outputs

# Main function to process all tasks and evaluate the outputs
async def main():
    # Load the test set
    test_set_path = 'test_set_processing/test_set.jsonl'
    test_cases = load_test_set(test_set_path)

    print(f"Loaded {len(test_cases)} test cases")

    print(f"{test_cases[0] = }")

    test_cases = [test_cases[0], test_cases[1]]
    
    # Process each task separately and collect outputs
    tasks = [
        ("idea_generation", idea_generation),
        ("method_recommendation", method_recommendation),
        ("outcome_prediction", outcome_prediction),
        ("future_work_recommendation", future_work_recommendation),
        ("predict_title", predict_title)
    ]
    
    results = {}
    
    for task_name, generate_prompt_fn in tasks:
        outputs = await process_task(task_name, generate_prompt_fn, test_cases)
        references = [entry[task_name] for entry in test_cases]
        metrics = compute_all_metrics(predictions=outputs, references=references)
        results[task_name] = metrics
    
    # Save the results
    output_path = 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation results saved to {output_path}")

# Run the main function
if __name__ == '__main__':
    asyncio.run(main())
