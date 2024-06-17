"""Compute cosine similarity, ROUGE, BLEU, METEOR, and BERTScore metrics.

See example usage in the `__main__` block at the end of the file.
"""

import json
import asyncio
from typing import List, Union

import evaluate
import numpy as np
from sentence_transformers import SentenceTransformer
from massw.models.gpt_azure import AzureConfig, Batch

LLM_SIM_PROMPT = """
You are an expert in Computer Science with a specialization in text analysis,
you are tasked to evaluate the semantic similarity between pairs of abstractive
summarizations from scientific papers. Each summarization pertains to the same
aspect (Context, Key Idea, Method, Outcome, or Projected Impact) of the same
paper.

For each pair of summarizations, classify the aspect,
and assign a similarity score on a scale from 1 to 10,
where 1 indicates completely dissimilar and 10
indicates identical content. Before scoring, include a brief justification for
your score.
You should output your results in JSON format as shown in the example.

Example Input:
Input 1: The experiments demonstrated a 20% increase in efficiency,
confirming the proposed model's effectiveness.
Input 2: Results show that the new model outperforms existing ones
by improving efficiency by approximately 20%.

Example JSON Output:
{
  "aspect": "Outcome",
  "score": 9,
  "justification": "Both texts describe similar measurable improvements
  in efficiency, closely aligning in their depiction
  of the model's effectiveness."
}
"""


class LLMSimilarity:
    """Evaluate the similarity between two texts using a language model."""

    def __init__(self, model_name: str = "gpt-4"):
        """Initialize the language model similarity evaluator."""
        assert model_name in ["gpt-4", "gpt-35-turbo"]
        self.model_name = model_name
        self.tpm = {"gpt-4": 4000, "gpt-35-turbo": 40000}[model_name]

    def generate_prompt(self, text_1: str, text_2: str):
        """Generate the prompt for the language model."""
        user_prompt = f"Input 1: {text_1}\nInput 2: {text_2}"
        messages = [
            {
                "role": "system",
                "content": LLM_SIM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        return messages

    async def _compute(self,
                       predictions: List[str],
                       references: List[List[str]]):
        """Compute the similarity between predictions and references."""
        batch = Batch(tpm=self.tpm, azure=AzureConfig())
        if isinstance(references[0], list):
            new_predictions = []
            for pred, refs in zip(predictions, references):
                new_predictions.extend([pred] * len(refs))
            new_references = [ref for refs in references for ref in refs]
            predictions, references = new_predictions, new_references
        # Compute similarity between each prediction and reference
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            messages = self.generate_prompt(pred, ref)
            await batch.add(endpoint="chat.completions.create",
                            response_format={"type": "json_object"},
                            model=self.model_name,
                            messages=messages,
                            metadata={"id": i})
        results = await batch.run()
        results = results.sort_values("id")
        outputs = results["result"].apply(
            lambda x: x["choices"][0]["message"]["content"])
        try:
            scores = outputs.apply(lambda x: float(json.loads(x)["score"]))
        except ValueError as e:
            print(e)
            scores = None
        scores = scores.dropna()
        scores = scores.to_numpy()
        try:
            scores = scores.reshape(len(predictions), -1)
            return {"llm_sim": float(np.mean(np.max(scores, axis=1)))}
        except ValueError as e:
            print(e)
            return {"llm_sim": float(np.mean(scores))}

    def compute(self, predictions: List[str], references: List[List[str]]):
        """Compute the similarity between predictions and references."""
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(
            self._compute(predictions, references))
        return results


class CosineSimilarity:
    """Compute cosine similarity between two ordered list of texts."""

    def __init__(self):
        """Initialize the SentenceTransformer model."""
        self.encoder = SentenceTransformer(
            'intfloat/multilingual-e5-large-instruct')

    def get_detailed_instruct(self, query: str) -> str:
        """Generate a detailed instruct for the query."""
        return f"Instruct: Retrieve semantically similar text.\nQuery: {query}"

    def get_embeddings(self, texts: List[str], is_query: bool):
        """Compute embeddings for the given texts."""
        if is_query:
            texts = [self.get_detailed_instruct(query) for query in texts]
        embeddings = self.encoder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ):
        """Compute the cosine similarity between predictions and references."""
        length = len(predictions)
        if isinstance(references[0], list):
            new_predictions = []
            for pred, refs in zip(predictions, references):
                new_predictions.extend([pred] * len(refs))
            new_references = [ref for refs in references for ref in refs]
            predictions, references = new_predictions, new_references
        predictions_embeddings = self.get_embeddings(predictions,
                                                     is_query=True)
        references_embeddings = self.get_embeddings(references, is_query=False)
        # Compute pairwise cosine similarity
        cosine_similarities = []
        for pred, ref in zip(predictions_embeddings, references_embeddings):
            cosine_similarities.append(np.dot(pred, ref))  # Already normalized
        cosine_similarities = np.array(cosine_similarities)
        cosine_similarities = cosine_similarities.reshape(length, -1)
        cosine_similarities = np.max(cosine_similarities, axis=1)
        return {"cosine": float(np.mean(cosine_similarities))}


class NAHit:
    """Compute the precision, recall, and F1 score for N/A hit metric."""

    def is_na(self, s: str):
        """Check if the string is N/A."""
        if s.lower() in ["n/a", "na", "not applicable"]:
            return True
        if len(s.split()) < 3:
            return True
        return False

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ):
        """Compute the precision, recall, and F1 score for N/A hit metric."""
        predictions_na = [self.is_na(pred) for pred in predictions]
        if isinstance(references[0], list):
            references_na = []
            references_na = [
                all(self.is_na(ref) for ref in refs) for refs in references
            ]
        else:
            references_na = [self.is_na(ref) for ref in references]
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        return {
            "precision":
            precision.compute(
                predictions=predictions_na,
                references=references_na,
            )["precision"],
            "recall":
            recall.compute(
                predictions=predictions_na,
                references=references_na,
            )["recall"],
            "f1":
            f1.compute(
                predictions=predictions_na,
                references=references_na,
            )["f1"],
            "pred_ratio":
            sum(predictions_na) / len(predictions_na),
            "ref_ratio":
            sum(references_na) / len(references_na),
        }


cs = CosineSimilarity()
llm_sim = LLMSimilarity()
bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt",
                       module_type="metric",
                       checkpoint="BLEURT-20-D12",
                       config_name="BLEURT-20-D12")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
nahit = NAHit()


def compute_metrics(predictions: List[str],
                    references: List[List[str]],
                    metric_names=None):
    """Compute cosine similarity, ROUGE, BLEU, METEOR, and BERTScore."""
    if metric_names is None:
        metric_names = [
            "cosine",
            "rouge",
            "bleu",
            "meteor",
            "bleurt",
            "bertscore",
            "nahit",
            "llm_sim"
        ]
    metrics = {}
    if "nahit" in metric_names:
        metrics["nahit"] = nahit.compute(
            predictions=predictions,
            references=references,
        )
    # Remove N/A predictions and references
    if isinstance(references[0], list):
        references_na = [
            all(nahit.is_na(ref) for ref in refs) for refs in references
        ]
    else:
        references_na = [nahit.is_na(ref) for ref in references]
    predictions_na = [nahit.is_na(pred) for pred in predictions]
    both_not_na = [
        not pred_na and not ref_na
        for pred_na, ref_na in zip(predictions_na, references_na)
    ]
    predictions = [
        pred for pred, not_na in zip(predictions, both_not_na) if not_na
    ]
    references = [
        ref for ref, not_na in zip(references, both_not_na) if not_na
    ]

    metric_computation_functions = {
        "cosine": cs,
        "llm_sim": llm_sim,
        "rouge": rouge,
        "bleu": bleu,
        "meteor": meteor,
        "bertscore": bertscore,
        "bleurt": bleurt
    }

    for metric_name in metric_names:
        if metric_name in metric_computation_functions:
            if metric_name == "bertscore":
                score = metric_computation_functions[metric_name].compute(
                    predictions=predictions,
                    references=references,
                    lang="en"
                )
                metrics[metric_name] = {
                    "precision": np.array(score["precision"]).mean(),
                    "recall": np.array(score["recall"]).mean(),
                    "f1": np.array(score["f1"]).mean()
                }
            elif metric_name == "bleurt":
                if isinstance(references[0], list):
                    grouped_references = list(zip(*references))
                    scores = []
                    for grouped_reference in grouped_references:
                        score = \
                            metric_computation_functions[metric_name].compute(
                                predictions=predictions,
                                references=grouped_reference)
                        scores.append(score["scores"])
                    scores = np.array(scores)  # (num_refs, num_preds)
                    score = np.mean(np.max(scores, axis=0))
                else:
                    score = metric_computation_functions[metric_name].compute(
                        predictions=predictions,
                        references=references)
                    score = np.mean(score["scores"])
                metrics[metric_name] = {"bleurt": score}
            else:
                metrics[metric_name] = \
                    metric_computation_functions[metric_name].compute(
                        predictions=predictions,
                        references=references
                )

    return metrics


def flatten_metrics(metric_dict: dict):
    """Flatten the metric dictionary for easy display."""
    flat_metrics = {}
    if "meteor" in metric_dict:
        flat_metrics["METEOR"] = metric_dict["meteor"]["meteor"]
    if "cosine" in metric_dict:
        flat_metrics["Cosine Embedding"] = metric_dict["cosine"]["cosine"]
    if "bleu" in metric_dict:
        flat_metrics["BLEU"] = metric_dict["bleu"]["bleu"]
        flat_metrics["Precision-1"] = metric_dict["bleu"]["precisions"][0]
        flat_metrics["Precision-2"] = metric_dict["bleu"]["precisions"][1]
        flat_metrics["Length Ratio"] = metric_dict["bleu"]["length_ratio"]
    if "rouge" in metric_dict:
        flat_metrics["ROUGE-1"] = metric_dict["rouge"]["rouge1"]
        flat_metrics["ROUGE-2"] = metric_dict["rouge"]["rouge2"]
    if "nahit" in metric_dict:
        flat_metrics["N/A-precision"] = metric_dict["nahit"]["precision"]
        flat_metrics["N/A-recall"] = metric_dict["nahit"]["recall"]
        flat_metrics["N/A-f1"] = metric_dict["nahit"]["f1"]
        flat_metrics["N/A in pred"] = metric_dict["nahit"]["pred_ratio"]
        flat_metrics["N/A in ref"] = metric_dict["nahit"]["ref_ratio"]
    if "bertscore" in metric_dict:
        flat_metrics["BERTScore-precision"] = metric_dict["bertscore"][
            "precision"]
        flat_metrics["BERTScore-recall"] = metric_dict["bertscore"]["recall"]
        flat_metrics["BERTScore-f1"] = metric_dict["bertscore"]["f1"]
    if "bleurt" in metric_dict:
        flat_metrics["BLEURT"] = metric_dict["bleurt"]["bleurt"]
    if "llm_sim" in metric_dict:
        flat_metrics["LLM Similarity"] = metric_dict["llm_sim"]["llm_sim"]
    return flat_metrics


if __name__ == "__main__":
    predictions_demo = ["The cat sat on the mat.", "The dog ate my homework."]
    references_demo = [["The cat sat on the mat.", "The cat sat on the desk."],
                       ["The dog ate my homework.", "The dog ate my lunch."]]

    # Compute metrics
    metrics_demo = compute_metrics(predictions=predictions_demo,
                                   references=references_demo,
                                   metric_names=["llm_sim"])

    # Print results
    print(json.dumps(metrics_demo, indent=2))
