# Automated Evaluation

The automated evaluation contains two parts. The evaluation for the most fitting embedding model 
and the evaluation of the whole RAG pipeline using the RAGAS framework.


## Embedding Model Evaluation
To evaluate different embedding models we adopted [Salha's apporach](https://github.com/MohamedBenSalha/confluence-rag). 
He calculates the query score which is a combination of F1@k, NDCG@k and a complexity score which is given to each query. 

$$
\text{Query Score} = \left( \frac{2}{3} \times \text{F1@K} + \frac{1}{3} \times \text{NDCG@K} \right) \times \text{Query Complexity}
$$

The Query score is computed for each query in the evaluation. The results will then be averaged for each embedding model.


To run the embedding model evaluation, follow these steps: 
1. Run the ``generate_results_with_relevance`` function by simply running ``embedding_eval.py``.  
2. Open ``results_with_relevance.csv`` and manually fill out the relevance field for each entry, 
by deciding if the retrieved context was relevant to the query. 
Relevance can be 0 if not relevant and 1 if relevant. 
3. Run the ``calc_metrics`` function from ``embedding_eval.py``.
   
The results then can be looked at in ``query_embedding_source_relevance_with_score.csv`` and ``embeddings_avarage_score.csv``.

## RAGAS
[RAGAS](https://docs.ragas.io/en/stable/) is a evaluation framework specifically designed to evaluate RAG implementations. 
For this specific RAG pipeline we run three core metrics of the framework. 

1. **Faithfulness**, which measures the extent to which the claims made in the generated response can be inferred from the retrieved context.
2. **Answer Relevance**, which assesses the degree to which the response directly and appropriately addresses the posed question.
3. **Context Precision**, which measures the proportion of retrieved chunks from the knowledge base that are relevant to a given query.


### Run with OpenAI API
If you have an OpenAI API key specified inside ``.env``, you can run the RAGAS evaluation by executing ``ragas_eval.py``.
This will use the default LLM and embedding model.

### Run with Different Evaluation Model
If you don't have an OpenAI API key you have to use different models for the evaluation. A simple way is to Ollama. 
Therefore you can uncomment the lines of code inside ``ragas_eval.py``, like discribed inside the file. 
Make sure you have the model you want to use downloaded and run locally. 

Note that using different models can cause parsing errors. This is why the OpenAI models are recommended. Not every model works with RAGAS!