# Evaluating Retrieval-Augmented Generation to AI for Requirements Engineering

This repository contains the code for implementing Retrieval-Augmented Generation (RAG) with RE-Model, a 7B parameter LLM fine-tuned for requirements engineering tasks. The implementation is designed to enhance RE-Model's performance by incorporating external knowledge retrieval during the generation process, aimed at improving accuracy and relevance for case-specific tasks.


## How to Use


### Requirements
Before you can run the pipeline make sure that the following requirements are fulfilled:

1. CUDA has to be installed. You can find the documentation [here](https://docs.nvidia.com/cuda/).
2. Ollama has to be installed You can find the documentation [here](https://github.com/ollama/ollama/blob/main/docs/README.md).
3. Make sure you created a virutal python environment and install all the dependencies. You can do that by executing the following commands:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Once this is done you can continue with the following steps.

### Quickstart
1. Download RE-Model by executing ``ollama run kasrahabib/zephyr-7b-kasra``
2. Run ``build_vectordatabase.py``
3. Run ``rag.py`` to query the RAG pipeline




### Data
The data of this RAG implementation contains two sources. 
First a [dronology requirements dataset](https://dronology.info/datasets/) 
and second a [unmanned aircraft system policy](https://www.caa.co.uk/our-work/publications/documents/content/cap-722/) 
from the United Kingdom. 
Both files can be found in the ``data`` folder.


### FAISS Vector Database
The next step is to fill the database, in our case we use FAISS, with the preprocessed data. 
In this step the data gets chunked using a fixed token size of 512 tokens, 
to match the embedding models maximum input sequence. By running ``build_vectordatabase.py`` the data will be 
chunked and then stored inside a new created FAISS database. 
The database will be stored locally. 

### Embedding Model
The RAG mechanism uses the ``thenlper/gte-base`` embedding model. 
If you were to use a different model, you can simply add it to ``load_embedding_model.py`` 
and change ``EMBEDDING_MODEL_NAME`` inside ``rag.py`` and ``build_vectordatabase.py``.
Then you have to build the vectordatabase again, before you can run the RAG pipeline,
using the respective python script.

### RE-Model
RE-Model is the generation component of our RAG pipeline. It is a zephyr-7b model, 
which was fine-tuned for the requirements engineering task. In this implementation the model is run locally.
Therefore it has to be downloaded first. To do this, Ollama has to be installed, since the model is hosted there. 
Once it is installed, you need to download the model via the following command ``ollama run kasrahabib/zephyr-7b-kasra``. 
If you were to change the generation model, 
simply download a new model from Ollama and change ``OLLAMA_MODEL_NAME`` inside ``rag.py``.


### Run the RAG Pipeline
Once all of the previous steps are done, you can run the RAG pipeline by running the ``rag.py`` script.
The query is specified in the main function of the script if you want to change it. 


## Evaluation
To evaluate this implementation we ran automated evaluations and conducted a user study, for which we calculated statistical tests.


### Automated Evaluation
More details to the embedding model and ragas evaluation can be found [here](evaluation/README.md).


### Statistical Tests
More details about the Wilcoxon test and the power analysis can be found [here](statistical_tests/README.md)