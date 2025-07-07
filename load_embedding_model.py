from langchain_huggingface import HuggingFaceEmbeddings

"""
Loads the embedding model used for generating vector representations of text data.
Feel free to add your customized embedding model.
"""


def load_embedding_model(model_name):
    print(f"Using model: {model_name}")

    model_configs = {
        "thenlper/gte-small": {},
        "jinaai/jina-embeddings-v2-base-de": {"trust_remote_code": True},
        "thenlper/gte-base": {},
        "BAAI/bge-small-en-v1.5": {},
        "intfloat/e5-small-v2": {},
    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        multi_process=False,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True},
    )
