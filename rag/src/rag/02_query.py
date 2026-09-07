#%%
import os
import dotenv

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))



from openai import OpenAI


dotenv.load_dotenv()


import qdrant_client
from qdrant_client import models


from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

#%%

mlflow.search_registered_models()
#%%

modelos = mlflow.search_registered_models(filter_string="name = 'rag3'")[0]
MODEL_GUARDRAILS_CUTOFF = modelos.tags["cutoff"]
MODEL_GUARDRAILS_NAME = modelos.name
MODEL_GUARDRAILS_LAST_VERSION = max([ int(v.version) for v in modelos.latest_versions])


MODEL_GUARDRAILS_URI = f"models:/{MODEL_GUARDRAILS_NAME}/{MODEL_GUARDRAILS_LAST_VERSION}"

#MODEL_GUARDRAILS_URI = "models:/novo_rag/1"
MODEL_GUARDRAILS = mlflow.sklearn.load_model(MODEL_GUARDRAILS_URI)
MODEL_GUARDRAILS


#%%

DENSE_MODEL = 'intfloat/multilingual-e5-large'
SPARSE_MODEL = 'Qdrant/BM25' 
COLBERT_MODEL = 'colbert-ir/colbertv2.0'

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



qdrant_client = qdrant_client.QdrantClient(
    url=QDRANT_CLUSTER_ENDPOINT,
    api_key=QDRANT_API_KEY,
)

client = OpenAI(
    api_key= GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
    
)



#%%

doc_convert = DocumentConverter()

chunker = HybridChunker(
    tokenizer=AutoTokenizer.from_pretrained(DENSE_MODEL),
    max_tokens=400,
)


dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)


while True:
    query = input("Entre com uma pergunta relacionada a introdução a machine learning: ")
    
    if query =="":
        break
    
    
    dense_query = list(dense_model.passage_embed(query))[0].tolist()
    
    
    prob_guardrails = MODEL_GUARDRAILS.predict_proba([dense_query])[0][1]
    if prob_guardrails < float(0.20):
        print("Pergunta fora do contexto. Tente Reformular")
        continue
    
    
    sparse_query = list(sparse_model.passage_embed(query))[0].as_object()
    colbert_query = list(colbert_model.passage_embed(query))[0].tolist()
    
    results = qdrant_client.query_points(
        collection_name="rag",
        prefetch={
            "prefetch": [
                {"query": dense_query, "using":"dense", "limit": 10},
                {"query": sparse_query, "using":"sparse", "limit": 10},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit":10,
        },
        query = colbert_query,
        using="colbert",
        limit=3,
    )

    for r in results.points:
        print(r.score, r.payload["text"])
    print("-----"*3)
    
    prompt = f"""Responda a seguinte pergunta usando os seguintes parágrafos de contexto:
    
    Pergunta: {query}
    
    Contexto:
    {'\n'.join([f'-{r.payload["text"]}\n' for r in results.points])}
    
    Responda no maximo com 100 caracteres
    """
    
    response = client.responses.create(
        input="Responda a seguinte pergunta",
        model="openai/gpt-oss-20b",
    )
print(response.output_text)



