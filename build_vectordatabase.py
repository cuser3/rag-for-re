from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from load_embedding_model import load_embedding_model
import json
from transformers import AutoTokenizer
import os


BASE_DIR = BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_NAME = "thenlper/gte-base"

SEPARATORS = [
    r"\n\d+(\.\d+)*\.\s",  # numbered sections 1., 1.1., 1.1.1., ...
    "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    " ",  # Spaces
    "",  # Split by characters if needed
]


def load_dronology_dataset(chunk_size=512, tokenizer_name=EMBEDDING_MODEL_NAME):
    """
    :return: List of LangchainDocuments
    """
    dronology_path = os.path.join(BASE_DIR, "data/dronology_dataset.json")
    with open(dronology_path, "r") as file:
        ds = json.load(file)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    raw_dataset = []
    current_chunk = ""
    current_metadata = {"issueid": [], "issuetype": [], "source": "dronology_dataset"}

    for entry in tqdm(ds["entries"]):
        entry_content = (
            entry["attributes"]["summary"] + "\n" + entry["attributes"]["description"]
        )
        entry_length = len(tokenizer.encode(entry_content))

        if len(tokenizer.encode(current_chunk)) + entry_length <= chunk_size:
            current_chunk += "\n\n" + entry_content
            current_metadata["issueid"].append(entry["issueid"])
            current_metadata["issuetype"].append(entry["attributes"]["issuetype"])
        else:
            raw_dataset.append(
                LangchainDocument(
                    page_content=current_chunk.strip(),
                    metadata=current_metadata,
                )
            )
            current_chunk = entry_content
            current_metadata = {
                "issueid": [entry["issueid"]],
                "issuetype": [entry["attributes"]["issuetype"]],
                "source": "dronology_dataset",
            }

    if current_chunk:
        raw_dataset.append(
            LangchainDocument(
                page_content=current_chunk.strip(),
                metadata=current_metadata,
            )
        )

    return raw_dataset


def load_uk_policy():
    """
    :return: A LangchainDocument wrapped inside a List to facilitate later implementation
    """
    policy_path = os.path.join(BASE_DIR, "data/ukpol.txt")
    with open(policy_path, "r") as file:
        uk_policy_content = file.read()

    raw_dataset = [
        LangchainDocument(
            page_content=uk_policy_content, metadata={"source": "UK_UAS_Policy"}
        )
    ]
    return raw_dataset


def split_docs(chunk_size, knowledge_base, tokenizer_name):
    """
    Split documents into chunks of maximum size `chunk_size` tokens or characters and return a list of unique documents.

    :param chunk_size: Maximum size of each chunk (in tokens or characters).
    :param knowledge_base: List of documents to be split.
    :param tokenizer_name: Tokenizer model to use for token-based splitting.
    :return: List of documents split into chunks.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=SEPARATORS,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)
    return docs_processed_unique


def build_database(embedding_name):
    dronology_docs_processed = split_docs(
        chunk_size=512,
        knowledge_base=load_dronology_dataset(),
        tokenizer_name=embedding_name,
    )

    uk_policy_docs_processed = split_docs(
        chunk_size=512,
        knowledge_base=load_uk_policy(),
        tokenizer_name=embedding_name,
    )

    all_docs_processed = dronology_docs_processed + uk_policy_docs_processed

    embedding_model = load_embedding_model(embedding_name)

    # Create the database
    VECTOR_DATABASE = FAISS.from_documents(
        all_docs_processed,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Save the database
    VECTOR_DATABASE.save_local(f"vectordatabase/{embedding_name}")
    print("Vector database saved!")


if __name__ == "__main__":
    build_database(EMBEDDING_MODEL_NAME)
