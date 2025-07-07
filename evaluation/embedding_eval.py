from langchain_community.vectorstores import FAISS
import numpy as np
import pandas as pd
import json
import os
import sys

sys.path.append("..")
from load_embedding_model import load_embedding_model
from build_vectordatabase import build_database

embeddings = [
    "thenlper/gte-small",
    "thenlper/gte-base",
    "jinaai/jina-embeddings-v2-base-de",
    "BAAI/bge-small-en-v1.5",
    "intfloat/e5-small-v2",
]  # Name of every embedding model to evaluate
TOP_K = 3
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def dcg_at_k(relevance_scores):
    print(relevance_scores)
    relevance_scores = np.asfarray(relevance_scores)[:TOP_K]
    if relevance_scores.size:
        return np.sum(
            (2**relevance_scores - 1) / np.log2(np.arange(2, relevance_scores.size + 2))
        )
    else:
        return 0.0


def ndcg_at_k(relevance_scores, number_relevant_sources):
    dcg_max_table = [1] * number_relevant_sources + [0] * (
        TOP_K - number_relevant_sources
    )
    if dcg_max_table is None:
        raise ValueError("dcg_max_table should be defined")
    dcg_max = dcg_at_k(dcg_max_table)
    return dcg_at_k(relevance_scores) / dcg_max


def generate_results_with_relevance(embeddings):
    results = []
    for embedding in embeddings:
        embedding_model = load_embedding_model(embedding)
        build_database(embedding)
        prompts = [
            {
                "query": "Example query",
                "complexity_score": 1,
                "number_relevant_sources": 3,  # number of documents that should be retrieved
            },  # ...
        ]
        KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
            f"{BASE_DIR}/vectordatabase/{embedding}",
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        for prompt in prompts:
            retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(
                prompt["query"], k=TOP_K
            )
            for doc in retrieved_docs:
                results.append(
                    {
                        "query": prompt["query"],
                        "embedding": embedding,
                        "complexity_score": prompt["complexity_score"],
                        "number_relevant_sources": prompt["number_relevant_sources"],
                        "source": doc.page_content.replace("\n", " "),
                    }
                )
    df = pd.DataFrame(results)
    df["relevance"] = ""

    columns_to_include = [
        "query",
        "embedding",
        "complexity_score",
        "number_relevant_sources",
        "source",
        "relevance",
    ]
    df.to_csv(
        "results_with_relevance.csv",
        columns=columns_to_include,
        encoding="utf-8",
        index=False,
    )
    print("Saved results to results_with_relevance.csv")


def calc_metrics():
    df = pd.read_csv("results_with_relevance.csv")
    metrics = (
        df.groupby(["query", "embedding", "complexity_score"])
        .apply(
            lambda group: pd.Series(
                {
                    "precision": group["relevance"].sum() / TOP_K,
                    "recall": group["relevance"].sum()
                    / group["number_relevant_sources"].iloc[0],
                    "NDCG": ndcg_at_k(
                        group["relevance"].tolist(),
                        group["number_relevant_sources"].iloc[0],
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    metrics["f1"] = (
        2
        * (metrics["precision"] * metrics["recall"])
        / (metrics["precision"] + metrics["recall"])
    )
    metrics["score"] = ((2 / 3) * metrics["f1"] + (1 / 3) * metrics["NDCG"]) * metrics[
        "complexity_score"
    ]
    metrics.to_csv("query_embedding_source_relevance_with_score.csv")

    # Group by 'embedding' and calculate the average of 'score'
    average_scores = metrics.groupby("embedding")["score"].mean().reset_index()
    # Rename the columns for clarity
    average_scores.columns = ["embedding", "average_score"]
    # Sort by average score descending
    average_scores = average_scores.sort_values(by="average_score", ascending=False)

    print("\nAverage Scores for Each Embedding:")
    print(average_scores)

    average_scores.to_csv("embeddings_average_scores.csv", index=False)


if __name__ == "__main__":
    # 1. Run generate_results_with_relevance()
    # 2. Manually fill out relevance field in results_with_relevance.csv
    # 3. Run calc_metrics()
    generate_results_with_relevance(embeddings)
    # calc_metrics()
