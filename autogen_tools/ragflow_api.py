from ragflow import RAGFlow
import os
os.environ["RAGFLOW_API_KEY"] = "<YOUR_API_KEY>"
os.environ["RAGFLOW_BASE_URL"] = "http://<YOUR_BASE_URL>:9380"
DATSET_IDS = [] # your dataset ids
def retrieve_by_ragflow(question:str)->str:
    rag_object = RAGFlow(api_key=os.environ["RAGFLOW_API_KEY"], base_url=os.environ["RAGFLOW_BASE_URL"])
    res = rag_object.retrieve(question=question,
                 dataset_ids=DATSET_IDS,
                 page=1, page_size=30, similarity_threshold=0.2,
                 vector_similarity_weight=0.3,
                 top_k=1024
                 )
    return str(res)
