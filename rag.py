from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from load_embedding_model import load_embedding_model
import ollama
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_MODEL_NAME = "thenlper/gte-base"
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vectordatabase", EMBEDDING_MODEL_NAME)
OLLAMA_MODEL_NAME = "kasrahabib/zephyr-7b-kasra"
PROMPT_TEMPLATE = """
        You are a professional requirements engineer.
        Using the information contained in the context, 
        give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question.
        Always provide the number of the source document from the context.
        Only provide the answer to the question.
        
        Context:
        {context}

        Question: {question}
        """


def query_ollama_model(prompt: str, model_name: str = OLLAMA_MODEL_NAME) -> str:
    """
    Query the Ollama model with a given prompt and return the generated response.
    """
    client = ollama.Client()
    response = client.generate(model=model_name, prompt=prompt)
    return response.response


def perfrom_rag(query, knowledgebase):
    KNOWLEDGE_VECTOR_DATABASE = knowledgebase

    # Perform RAG workflow
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query, k=3)
    retrieved_contexts = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in retrieved_docs
    ]
    context = "\n".join(
        [
            f"-----------------------Document {str(i)}: Source: {doc.metadata['source']}-------------------\n"
            + doc.page_content
            for i, doc in enumerate(retrieved_docs)
        ]
    )

    # Format the prompt template and add context and query
    chat_prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    final_prompt = chat_prompt_template.format(question=query, context=context)
    ollama_answer = query_ollama_model(final_prompt, OLLAMA_MODEL_NAME)

    print("Prompt:\n" + final_prompt)
    print("RAG RESPONSE: \n" + ollama_answer)

    response_dict = {
        "llm": OLLAMA_MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "query": query,
        "response": ollama_answer,
        "context": retrieved_contexts,
    }

    return response_dict


if __name__ == "__main__":
    query = """
        If a UAV's battery level drops below the threshold, what emergency procedures are triggered?
        """
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
        VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    perfrom_rag(query, KNOWLEDGE_VECTOR_DATABASE)
