"""Api module for interacting with the Mistral model."""
import urllib.request
import json
import pandas as pd
import time
import os
from typing import Dict


def prompts_to_raw_output_mistral(messages):
    """Process prompts using the specified Mistral model endpoint."""
    final_results = pd.DataFrame(columns=['pid', 'output'])

    url = os.environ.get("MISTRAL_API_URL")
    api_key = os.environ.get("MISTRAL_API_KEY")
    print(f"Using Mistral API at {url}")
    print(f"Using Mistral API key: {api_key}")
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'azureml-model-deployment': 'mistralai-mixtral-8x7b-instru-7'
    }

    for pid, msg in messages:
        response_df = handle_mistral_model(url, headers, msg, {"pid": pid})
        final_results = pd.concat([final_results,
                                   response_df], ignore_index=True)

    return final_results


def raw_output_to_dict_mistral(output_path: str) -> Dict[str, str]:
    """
    Load and convert raw output from the Mistral model into a dictionary.

    Args:
        output_path (str): Path to the model output CSV file.

    Returns:
        Dict[str, str]: A dictionary mapping pid to processed output.
    """
    output_dict = {}
    task_output = pd.read_csv(output_path, sep="\t",
                              converters={'result': lambda x: json.loads(x)
                                          if x else None})
    for _, row in task_output.iterrows():
        output = row["output"]
        output_dict[row['pid']] = output
    return output_dict


def handle_mistral_model(url, headers, messages, entry):
    """Handle the Mistral model API request."""
    output_df = pd.DataFrame(columns=['pid', 'output'])
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            data = {
                "input_data": {
                    "input_string": messages,
                    "parameters": {
                        "temperature": 0,
                        "top_p": 0.9,
                        "do_sample": True,
                        "max_new_tokens": 200,
                        "return_full_text": True
                    }
                }
            }
            body = str.encode(json.dumps(data))
            req = urllib.request.Request(url, body, headers)
            print(f"{req = }")
            with urllib.request.urlopen(req) as response:
                result_json = json.loads(response.read())
                output_df = output_df.append({"pid": entry["pid"],
                                             "output": result_json},
                                             ignore_index=True)
            break
        except urllib.error.HTTPError as error:
            print(f"The request failed with status code: {error.code}")
            retries += 1
            time.sleep(2)
        # mistral has a werid excetion, need to change below
        # To avoid "Catching too general exception".
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            retries += 1
            time.sleep(2)

    return output_df
