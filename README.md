# Job Embedding Search API

This is a FastAPI-based application that performs job similarity search using embeddings from multiple models like GloVe, SpaCy, and Sentence-Transformers. It also evaluates and compares the performance of these models based on various metrics such as Precision, Recall, MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain). 

## Features
- **Job Similarity Search**: Search for similar jobs based on the job title and get the most relevant results.
- **Evaluation Metrics**: The API computes and provides several evaluation metrics, including Precision@5, Recall@5, MRR, and NDCG, to evaluate the performance of different embedding models.
- **Plotting Results**: The API generates a bar plot comparing the evaluation metrics across different embedding models.

## Requirements
- Python 3.7+
- FastAPI
- Uvicorn (for serving the FastAPI app)
- Faiss
- pandas
- numpy
- scikit-learn
- tqdm
- matplotlib

You can install the required dependencies by running:

```bash
pip install -r requirements.txt

