import numpy as np
import faiss
import ast
from tqdm import tqdm
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
import io
from fastapi.responses import StreamingResponse

# FastAPI instance
app = FastAPI()

# Pydantic model for job title search input
class JobSearchRequest(BaseModel):
    job_title: str
    top_k: int = Query(..., le=10, ge=1)

class JobEmbeddingSearch:
    def __init__(self, embedding_files):
        self.embedding_files = embedding_files
        self.results = defaultdict(dict)
        self.eval_df = None
    
    def load_and_process_embeddings(self, file_path, column_name):
        df = pd.read_csv(file_path)
        
        def parse_embedding(embedding_str):
            try:
                embedding_list = ast.literal_eval(embedding_str)
                return np.array(embedding_list, dtype=np.float32)
            except:
                try:
                    embedding_list = [float(x) for x in embedding_str.strip('[]').split()]
                    return np.array(embedding_list, dtype=np.float32)
                except Exception as e:
                    return np.zeros(384 if 'sentence' in file_path.lower() else 300, dtype=np.float32)

        tqdm.pandas(desc=f"Parsing embeddings ({file_path})")
        df[column_name] = df[column_name].progress_apply(parse_embedding)
        
        embedding_matrix = np.vstack(df[column_name])
        faiss.normalize_L2(embedding_matrix)
        return df, embedding_matrix

    def build_faiss_index(self, embedding_matrix):
        d = embedding_matrix.shape[1]
        nlist = 100
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embedding_matrix)
        index.add(embedding_matrix)
        index.nprobe = 10
        return index

    def find_similar_jobs(self, job_title, df, embedding_matrix, faiss_index, top_k):
        query_idx = df[df['jobTitle'] == job_title].index
        if query_idx.empty:
            return None, None
        query_idx = query_idx[0]
        query_embedding = embedding_matrix[query_idx].reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, top_k + 1)
        similar_indices = indices[0][1:]
        return df.iloc[similar_indices], distances[0][1:]

    def compute_metrics(self, job_title, similar_jobs, similarities, df, top_k):
        query_keywords = set(job_title.lower().split())
        relevant_jobs = df[df['jobTitle'].str.lower().apply(
            lambda x: any(keyword in x for keyword in query_keywords if len(keyword) > 3)
        )].index.tolist()
        
        retrieved_indices = similar_jobs.index.tolist()
        relevant_retrieved = len(set(retrieved_indices[:5]).intersection(relevant_jobs))
        precision_at_5 = relevant_retrieved / 5 if len(retrieved_indices) >= 5 else 0
        recall_at_5 = relevant_retrieved / len(relevant_jobs) if len(relevant_jobs) > 0 else 0
        
        mrr = 0
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_jobs:
                mrr = 1 / rank
                break
        
        true_relevance = [1 if idx in relevant_jobs else 0 for idx in retrieved_indices[:5]]
        if len(true_relevance) < 5:
            true_relevance.extend([0] * (5 - len(true_relevance)))
        ideal_relevance = sorted(true_relevance, reverse=True)
        ndcg = ndcg_score([ideal_relevance], [true_relevance]) if sum(true_relevance) > 0 else 0
        
        return {
            'Precision@5': precision_at_5,
            'Recall@5': recall_at_5,
            'MRR': mrr,
            'NDCG': ndcg
        }

    def run_search(self, job_title, top_k):
        for model, (file_path, column_name) in self.embedding_files.items():
            df, embedding_matrix = self.load_and_process_embeddings(file_path, column_name)
            index = self.build_faiss_index(embedding_matrix)
            similar_jobs, similarities = self.find_similar_jobs(job_title, df, embedding_matrix, index, top_k=top_k)
            
            if similar_jobs is not None:
                self.results[model]['similar_jobs'] = similar_jobs
                self.results[model]['similarities'] = similarities
                
                metrics = self.compute_metrics(job_title, similar_jobs, similarities, df, top_k)
                for metric, value in metrics.items():
                    self.results[model][metric] = value
        
        metrics = ['Precision@5', 'Recall@5', 'MRR', 'NDCG']
        self.eval_df = pd.DataFrame.from_dict(
            {model: [results[metric] for metric in metrics] for model, results in self.results.items()},
            orient='index',
            columns=metrics
        )

    def plot_results(self):
        if self.eval_df is None:
            return None
        
        self.eval_df.plot(kind='bar', figsize=(10, 6))
        plt.title("Comparison of Embedding Models Based on Evaluation Metrics")
        plt.ylabel("Score")
        plt.xlabel("Embedding Models")
        plt.xticks(rotation=45)
        plt.legend(title="Metrics")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        img_stream = io.BytesIO()
        plt.savefig(img_stream, format="png")
        img_stream.seek(0)
        return img_stream

# Define file paths and embedding column names
embedding_files = {
    'GloVe': ('jobs_with_glove_embeddings.csv', 'glove_embeddings'),
    'SpaCy': ('jobs_with_spacy_embeddings.csv', 'spacy_embeddings'),
    'Sentence-Transformers': ('jobs_with_sentence_embeddings.csv', 'embeddings')
}

# Initialize the JobEmbeddingSearch class
job_search = JobEmbeddingSearch(embedding_files)

@app.post("/search/")
async def search_job(request: JobSearchRequest):
    job_title = request.job_title
    top_k = request.top_k
    
    # Perform job search and calculate metrics
    job_search.run_search(job_title, top_k)
    
    # Return results as a dictionary
    return job_search.results

@app.get("/plot/")
async def plot_results():
    img_stream = job_search.plot_results()
    if img_stream is None:
        raise HTTPException(status_code=404, detail="No results available to plot.")
    return StreamingResponse(img_stream, media_type="image/png")

