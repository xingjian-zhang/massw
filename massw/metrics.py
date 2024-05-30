"""Compute cosine similarity, ROUGE, BLEU, METEOR, and BERTScore metrics.

See example usage in the `__main__` block at the end of the file.
"""

import asyncio
import json
from typing import List, Union

import api
import evaluate
import numpy as np


async def get_embeddings_helper(texts: List[str]):
    """Asynchronously get embeddings for a list of texts."""
    batch = api.Batch(tpm=200_000, azure=api.AzureConfig(), loglevel=0)
    for i, text in enumerate(texts):
        await batch.add(
            endpoint="embeddings.create",
            model="text-embedding-api",
            metadata={"id": i},
            input=text,
        )
    results_df = await batch.run()
    return results_df


class CosineSimilarity:
    """Compute cosine similarity between two ordered list of texts."""

    def get_embeddings(self, texts: List[str]):
        loop = asyncio.get_event_loop()
        results_df = loop.run_until_complete(get_embeddings_helper(texts))
        embeddings = results_df.sort_values("id").apply(
            lambda x: x["result"]["data"][0]["embedding"], axis=1)
        embeddings = np.array(embeddings.tolist())
        return embeddings

    def compute(
        self,
        predictions: List[str],
        references: Union[List[str], List[List[str]]],
    ):
        if isinstance(references[0], list):
            new_predictions = []
            for pred, refs in zip(predictions, references):
                new_predictions.extend([pred] * len(refs))
            new_references = [ref for refs in references for ref in refs]
            predictions, references = new_predictions, new_references
        predictions_embeddings = self.get_embeddings(predictions)
        references_embeddings = self.get_embeddings(references)
        # Compute pairwise cosine similarity
        cosine_similarities = []
        for pred, ref in zip(predictions_embeddings, references_embeddings):
            cosine_similarities.append(np.dot(pred, ref))  # Already normalized
        return {"cosine": np.mean(np.array(cosine_similarities))}


def compute_all_metrics(predictions: List[str], references: List[List[str]]):
    cs = CosineSimilarity()
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")
    metrics = {
        "cosine":
        cs.compute(
            predictions=predictions,
            references=references,
        ),
        "rouge":
        rouge.compute(
            predictions=predictions,
            references=references,
        ),
        "bleu":
        bleu.compute(
            predictions=predictions,
            references=references,
        ),
        "meteor":
        meteor.compute(
            predictions=predictions,
            references=references,
        ),
        "bertscore":
        bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
        ),
    }
    return metrics


if __name__ == "__main__":
    # Load demo data
    predictions = ["The cat sat on the mat.", "The dog ate my homework."]
    references = [["The cat sat on the mat.", "The cat sat on the desk."],
                  ["The dog ate my homework."]]

    # Compute metrics
    metrics = compute_all_metrics(predictions=predictions,
                                  references=references)

    # Print results
    print(json.dumps(metrics, indent=2))
    #
