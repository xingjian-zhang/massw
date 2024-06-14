"""
This program defines a series of functions that automate the generation.

Each function corresponds to a specific aspect of research paper generation, \
such as idea generation, method\
recommendation, outcome prediction, future work recommendation, \
and title prediction.
"""

SYSTEM_PROMPT = """
You are an expert in research tasked with generating detailed prompts for \
    various aspects of academic research papers.
Each task involves creating a specific type of prompt \
based on the provided information. Here are the definitions of \
each part you will work with:

- Concept
  - Definition
  - Relative Time

- Context: The status quo of related literature or reality \
which motivated this study.
This could normally be a problem, a research question, or a research gap \
that has not been successfully addressed by previous work. \
This is anything that happened before this study.

- Key Idea: The main intellectual merit of this paper, \
often in comparison to the context.
This could normally be a novel idea or solution proposed in this paper \
that distinguishes it from what's already done in literature.
This is proposed in this study.

- Method: The specific research method that investigates \
and validates the key idea.
This could be an experimental setup, a theoretical framework, or \
other necessary methodology to implement and/or evaluate the key idea.
This is performed in this study.

- Outcome: The factual statement about the study output.
This could be the experiment results and any other measurable \
outcome that has occurred.
It marks whether the key hypothesis is testified or not. \
This is produced in this study.

- Projected Impact: The author-anticipated impact of the work on the field, \
and potential further research identified by the author \
that may improve or extend this study.
This is anything being anticipated but has not happened yet.
"""


def idea_generation(data):
    """
    Generate a prompt for idea generation based on the provided context.

    Args:
        data (dict): Contains 'context' and 'key_idea' from the research data.

    Returns:
        tuple: A tuple containing the prompt and the ground truth for
        idea generation.
    """
    context = data['context']
    prompt = f"Given the context: '{context}', generate key ideas \
        that could advance this area of study. "
    ground_truth = data['key_idea']
    return prompt, ground_truth


def method_recommendation(data):
    """
    Recommend a method to validate a key idea.

    Args:
        data (dict): Contains 'context', 'key_idea', and 'method'
        from the research data.

    Returns:
        tuple: A tuple containing the prompt and the ground truth
        for method recommendation.
    """
    context = data['context']
    key_idea = data['key_idea']
    prompt = f"Given the context: '{context}' and the key idea: '{key_idea}', \
        recommend the most suitable method to validate this idea. "
    ground_truth = data['method']
    return prompt, ground_truth


def outcome_prediction(data):
    """
    Predict the potential outcome of a research.

    Args:
        data (dict): Contains 'context', 'key_idea', 'method', and 'outcome'.

    Returns:
        tuple: A tuple containing the prompt and the ground truth
        for outcome prediction.
    """
    context = data['context']
    key_idea = data['key_idea']
    method = data['method']
    prompt = f"Based on the context: '{context}', the key idea: '{key_idea}', \
        and the recommended method: '{method}', \
        predict the potential outcome of this research. "
    ground_truth = data['outcome']
    return prompt, ground_truth


def future_work_recommendation(data):
    """
    Suggest projected impact for the research.

    Args:
        data (dict): Contains 'context', 'key_idea', 'method', 'outcome', \
            and 'future_impact' from the research data.

    Returns:
        tuple: A tuple containing the prompt and the ground truth
        for future work.
    """
    context = data['context']
    key_idea = data['key_idea']
    method = data['method']
    outcome = data['outcome']
    prompt = f"Based on the context: '{context}', the key idea: '{key_idea}', \
        the method: '{method}', and the outcome: '{outcome}', \
            suggest projected impact for this research."
    ground_truth = data.get('future_impact', '')
    return prompt, ground_truth


def predict_title(data):
    """
    Predict the title of a research paper.

    Args:
        data (dict): Contains all necessary information from the research data.

    Returns:
        tuple: A tuple containing the prompt and the ground trut
        for title prediction.
    """
    context = data['context']
    key_idea = data['key_idea']
    method = data['method']
    outcome = data['outcome']
    future_impact = data['future_impact']
    prompt = f"Given the context: '{context}', the key idea: '{key_idea}', \
        the method: '{method}', the outcome: '{outcome}', \
        and the future impact: '{future_impact}', \
        predict the title of this research paper. \
        The title should be concise and reflective of the core aspects."
    ground_truth = data.get('title', '')
    return prompt, ground_truth
