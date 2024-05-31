import json

system_prompt = """
You are an expert in research tasked with generating detailed prompts for various aspects of academic research papers. Each task involves creating a specific type of prompt based on the provided information. Here are the definitions of each part you will work with:

- Concept
  - Definition
  - Relative Time

- Context: The status quo of related literature or reality which motivated this study. This could normally be a problem, a research question, or a research gap that has not been successfully addressed by previous work. This is anything that happened before this study.
  
- Key Idea: The main intellectual merit of this paper, often in comparison to the context. This could normally be a novel idea or solution proposed in this paper that distinguishes it from what’s already done in literature. This is proposed in this study.
  
- Method: The specific research method that investigates and validates the key idea. This could be an experimental setup, a theoretical framework, or other necessary methodology to implement and/or evaluate the key idea. This is performed in this study.
  
- Outcome: The factual statement about the study output. This could be the experiment results and any other measurable outcome that has occurred. It marks whether the key hypothesis is testified or not. This is produced in this study.
  
- Projected Impact: The author-anticipated impact of the work on the field, and potential further research identified by the author that may improve or extend this study. This is anything being anticipated but has not happened yet.
"""


# Load the test set
def load_test_set(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

# Task 1: Idea generation
def idea_generation(data):
    context = data['context']
    prompt = f"Given the context of '{context}', generate key ideas that could advance this area of study. "
    ground_truth = data['key_idea']
    assert ground_truth is not '', "Key Idea is missing in the data"
    return prompt, ground_truth

# Task 2: Method recommendation
def method_recommendation(data):
    context = data['context']
    key_idea = data['key_idea']
    prompt = f"Given the context: '{context}' and the key idea: '{key_idea}', recommend the most suitable method to validate this idea. "
    ground_truth = data.get('method', '')
    assert ground_truth != '', "Method is missing in the data"
    return prompt, ground_truth

# Task 3: Outcome prediction
def outcome_prediction(data):
    context = data['context']
    key_idea = data['key_idea']
    method = data['method']
    prompt = f"Based on the context: '{context}', the key idea: '{key_idea}', and the recommended method: '{method}', predict the potential outcome of this research. "
    ground_truth = data['outcome']
    assert ground_truth != '', "Outcome is missing in the data"
    return prompt, ground_truth

# Task 4: Future work recommendation/impact prediction
def future_work_recommendation(data):
    context = data['context']
    key_idea = data['key_idea']
    method = data['method']
    outcome = data['outcome']
    prompt = f"Based on the context: '{context}', the key idea: '{key_idea}', the method: '{method}', and the outcome: '{outcome}', suggest projected Impact for this research."
    ground_truth = data.get('future_impact', '')
    assert ground_truth != '', "Future Impact is missing in the data"
    return prompt, ground_truth

# Task 5: Predicting title based on all aspects
def predict_title(data):
    context = data['context']
    key_idea = data['key_idea']
    method = data['method']
    outcome = data['outcome']
    future_impact = data['future_impact']
    prompt = f"Given the context: '{context}', the key idea: '{key_idea}', the method: '{method}', the outcome: '{outcome}', and the future impact: '{future_impact}', predict the title of this research paper. The title should be concise and reflective of the core aspects."
    ground_truth = data.get('title', '')
    assert ground_truth != '', "Title is missing in the data"
    return prompt, ground_truth

# # Process the test set
# def process_test_set(test_set):
#     results = []
#     for entry in test_set:
#         context = entry['context']
#         key_idea = entry['key_idea']
#         method = entry.get('method', 'N/A')
#         outcome = entry.get('outcome', 'N/A')
#         future_impact = entry.get('future_impact', 'N/A')

#         generated_idea_prompt = idea_generation(context)
#         recommended_method_prompt = method_recommendation(context, key_idea)
#         predicted_outcome_prompt = outcome_prediction(context, key_idea, method)
#         recommended_future_work_prompt = future_work_recommendation(context, key_idea, method, outcome)
#         predicted_title_prompt = predict_title(context, key_idea, method, outcome, future_impact)

#         results.append({
#             "pid": entry['pid'],
#             "generated_idea_prompt": generated_idea_prompt,
#             "recommended_method_prompt": recommended_method_prompt,
#             "predicted_outcome_prompt": predicted_outcome_prompt,
#             "recommended_future_work_prompt": recommended_future_work_prompt,
#             "predicted_title_prompt": predicted_title_prompt
#         })
#     return results

# Main function to execute the testing process
# def main():
#     test_set_path = 'test_set_processing/test_set.jsonl'
#     test_set = load_test_set(test_set_path)
#     results = process_test_set(test_set)
    
#     # Output the results
#     for result in results:
#         print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     main()
