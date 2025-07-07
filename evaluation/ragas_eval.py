from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
)
from ragas import EvaluationDataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
from dotenv import load_dotenv


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

os.environ["OPENAI_API_KEY"] = openai_api_key


results = [
    {
        "user_input": "Example query",
        "response": "Example response",
        "retrieved_contexts": ["Context 1", "Context 2", "Context 3"],
    },  # ...
]

evaluation_dataset = EvaluationDataset(results)

# If you dont have an OpenAI API key, you can use the Ollama LLMs instead.
# Make sure the models are downloaded and running locally.
# Uncomment the commented lines in the evaluate function below to use them.
langchain_llm = ChatOllama(model="llama3.1")
langchain_embeddings = OllamaEmbeddings(model="llama3.1")

result = evaluate(
    evaluation_dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ],
    # llm=langchain_llm,
    # embeddings=langchain_embeddings
)

print(result)
