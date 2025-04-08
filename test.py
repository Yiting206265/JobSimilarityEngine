import numpy as np
import faiss
import ast
from tqdm import tqdm
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import Dict, List, Optional
from sklearn.metrics import ndcg_score
import re
import os
from sqlalchemy import create_engine, Column, String, LargeBinary, MetaData, Table, inspect, text, exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import time
import io
import csv
import logging
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI()

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://jobsearch:jobsearchpass@db:5432/jobsearchdb")
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
    logger.info("Database engine created successfully")
except Exception as e:
    logger.error(f"Failed to create database engine: {str(e)}")
    raise

# Pydantic model for job title search input
class JobSearchRequest(BaseModel):
    job_title: str
    top_k: int = Query(..., le=10, ge=1)

class JobEmbeddingSearch:
    def __init__(self, embedding_files: Dict[str, tuple]):
        self.embedding_files = embedding_files
        self.results = defaultdict(dict)
        self._wait_for_db()
        self._init_database()
    
    def _wait_for_db(self):
        """Wait for the database to be ready"""
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Database connection successful")
                return
            except Exception as e:
                logger.warning(f"Waiting for database... ({retry_count + 1}/{max_retries})")
                time.sleep(2)
                retry_count += 1
        raise Exception("Could not connect to database after multiple attempts")
    
    def _get_table_name(self, model_name: str) -> str:
        """Convert model name to SQLite-friendly table name"""
        table_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name.lower())
        if not table_name[0].isalpha():
            table_name = 't_' + table_name
        return table_name
    
    def _init_database(self):
        """Initialize the database and create necessary tables"""
        try:
            inspector = inspect(engine)
            
            # Create tables for each embedding model
            for model in self.embedding_files.keys():
                table_name = self._get_table_name(model)
                full_table_name = f"{table_name}_embeddings"
                
                # Check if table exists
                if not inspector.has_table(full_table_name):
                    logger.info(f"Creating table {full_table_name}")
                    metadata = MetaData()
                    Table(
                        full_table_name,
                        metadata,
                        Column('job_title', String, primary_key=True),
                        Column('embedding', LargeBinary)
                    )
                    metadata.create_all(engine)
                    
                    # Create index on job_title
                    with engine.connect() as conn:
                        conn.execute(text(f"CREATE INDEX idx_{table_name}_job_title ON {full_table_name} (job_title)"))
                        conn.commit()
                    logger.info(f"Created table and index for {full_table_name}")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _safe_read_csv(self, file_path: str, chunksize: int = 100000) -> pd.DataFrame:
        """Safely read large CSV files in chunks"""
        logger.info(f"Reading {file_path} in chunks of {chunksize} rows")
        chunks = []
        total_rows = 0
        
        try:
            # First try to read just the header to check column names
            with open(file_path, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                logger.info(f"CSV header: {header}")
            
            # Now read the file in chunks
            for chunk in pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', chunksize=chunksize):
                logger.info(f"Chunk columns: {chunk.columns.tolist()}")
                logger.info(f"Sample row: {chunk.iloc[0].to_dict() if len(chunk) > 0 else 'Empty chunk'}")
                
                chunks.append(chunk)
                total_rows += len(chunk)
                logger.info(f"Read {total_rows} rows, memory usage: {self._get_memory_usage():.2f} MB")
                
                # Clear memory if usage is high
                if self._get_memory_usage() > 1000:  # If using more than 1GB
                    gc.collect()
        except UnicodeDecodeError:
            logger.info("UTF-8 failed, trying latin1 encoding")
            for chunk in pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip', chunksize=chunksize):
                logger.info(f"Chunk columns: {chunk.columns.tolist()}")
                logger.info(f"Sample row: {chunk.iloc[0].to_dict() if len(chunk) > 0 else 'Empty chunk'}")
                
                chunks.append(chunk)
                total_rows += len(chunk)
                logger.info(f"Read {total_rows} rows, memory usage: {self._get_memory_usage():.2f} MB")
                
                if self._get_memory_usage() > 1000:
                    gc.collect()
        
        if not chunks:
            raise ValueError(f"No data could be read from {file_path}")
        
        logger.info(f"Concatenating {len(chunks)} chunks")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Final dataframe shape: {df.shape}, memory usage: {self._get_memory_usage():.2f} MB")
        return df
    
    def _validate_embeddings_file(self, df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """Validate embeddings and return cleaned dataframe"""
        logger.info(f"Validating dataframe with columns: {df.columns.tolist()}")
        
        # Check for required columns
        if 'jobTitle' not in df.columns:
            raise ValueError("Column 'jobTitle' not found")
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")
        
        # Check for empty values
        empty_titles = df['jobTitle'].isna().sum()
        empty_embeddings = df[column_name].isna().sum()
        
        logger.info(f"Found {empty_titles} empty job titles and {empty_embeddings} empty embeddings")
        
        if empty_titles > 0 or empty_embeddings > 0:
            logger.warning(f"Found {empty_titles} empty job titles and {empty_embeddings} empty embeddings")
            df = df.dropna(subset=['jobTitle', column_name])
        
        # Check for duplicate job titles
        duplicates = df['jobTitle'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate job titles")
            df = df.drop_duplicates(subset=['jobTitle'])
        
        # Log sample of the data
        logger.info(f"Sample job title: {df['jobTitle'].iloc[0] if len(df) > 0 else 'No data'}")
        logger.info(f"Sample embedding type: {type(df[column_name].iloc[0]) if len(df) > 0 else 'No data'}")
        logger.info(f"Sample embedding length: {len(df[column_name].iloc[0]) if len(df) > 0 and isinstance(df[column_name].iloc[0], (list, np.ndarray)) else 'N/A'}")
        
        return df
    
    def _load_embeddings_to_db(self):
        """Load embeddings from CSV files into the database"""
        session = SessionLocal()
        try:
            for model, (file_path, column_name) in self.embedding_files.items():
                table_name = self._get_table_name(model)
                full_table_name = f"{table_name}_embeddings"
                
                logger.info(f"Processing {model} with file {file_path} and column {column_name}")
                
                # Check if data already exists
                if session.execute(
                    text(f"SELECT EXISTS (SELECT 1 FROM {full_table_name} LIMIT 1)")
                ).scalar():
                    logger.info(f"Data for {model} already exists in database")
                    count = session.execute(
                        text(f"SELECT COUNT(*) FROM {full_table_name}")
                    ).scalar()
                    logger.info(f"Found {count} records in {full_table_name}")
                    continue
                
                logger.info(f"Loading {model} embeddings into database...")
                try:
                    # Read and validate the CSV file
                    logger.info(f"Reading CSV file: {file_path}")
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
                    logger.info(f"CSV file loaded with {len(df)} rows")
                    logger.info(f"Columns: {df.columns.tolist()}")
                    
                    # Check if required columns exist
                    if 'jobTitle' not in df.columns:
                        raise ValueError(f"Column 'jobTitle' not found in {file_path}")
                    if column_name not in df.columns:
                        raise ValueError(f"Column '{column_name}' not found in {file_path}")
                    
                    # Check for empty values
                    empty_titles = df['jobTitle'].isna().sum()
                    empty_embeddings = df[column_name].isna().sum()
                    logger.info(f"Found {empty_titles} empty job titles and {empty_embeddings} empty embeddings")
                    
                    if empty_titles > 0 or empty_embeddings > 0:
                        logger.warning(f"Removing rows with empty values")
                        df = df.dropna(subset=['jobTitle', column_name])
                    
                    # Check for duplicate job titles
                    duplicates = df['jobTitle'].duplicated().sum()
                    if duplicates > 0:
                        logger.warning(f"Found {duplicates} duplicate job titles")
                        df = df.drop_duplicates(subset=['jobTitle'])
                    
                    if len(df) == 0:
                        raise ValueError("No valid data after cleaning")
                    
                    # Process embeddings
                    logger.info("Processing embeddings...")
                    def safe_parse_embedding(embedding_str):
                        try:
                            if isinstance(embedding_str, str):
                                try:
                                    embedding_list = ast.literal_eval(embedding_str)
                                except:
                                    embedding_list = [float(x) for x in embedding_str.strip('[]').split()]
                            else:
                                embedding_list = embedding_str
                            
                            embedding = np.array(embedding_list, dtype=np.float32)
                            if np.isnan(embedding).any():
                                logger.warning("Found NaN values in embedding")
                                return None
                            return embedding
                        except Exception as e:
                            logger.warning(f"Failed to parse embedding: {str(e)}")
                            return None
                    
                    # Process embeddings with progress bar
                    tqdm.pandas(desc=f"Processing embeddings for {model}")
                    df['processed_embedding'] = df[column_name].progress_apply(safe_parse_embedding)
                    
                    # Remove rows with failed embeddings
                    failed_embeddings = df['processed_embedding'].isna().sum()
                    if failed_embeddings > 0:
                        logger.warning(f"Failed to parse {failed_embeddings} embeddings")
                        df = df.dropna(subset=['processed_embedding'])
                    
                    if len(df) == 0:
                        raise ValueError("No valid embeddings after processing")
                    
                    # Create embedding matrix
                    logger.info("Creating embedding matrix")
                    embedding_matrix = np.vstack(df['processed_embedding'].values)
                    logger.info(f"Processed embeddings with shape {embedding_matrix.shape}")
                    
                    # Build FAISS index
                    try:
                        d = embedding_matrix.shape[1]
                        nlist = min(100, len(df) // 10)
                        quantizer = faiss.IndexFlatIP(d)
                        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                        index.train(embedding_matrix)
                        index.add(embedding_matrix)
                        index.nprobe = min(10, nlist // 10)
                        
                        # Save FAISS index to file
                        os.makedirs('data', exist_ok=True)
                        faiss.write_index(index, f"data/{table_name}_index.faiss")
                        logger.info(f"Saved FAISS index for {model}")
                    except Exception as e:
                        logger.error(f"Error building FAISS index for {model}: {str(e)}")
                        continue
                    
                    # Write data to database
                    logger.info("Writing data to database...")
                    chunk_size = 1000  # Smaller chunk size for better error handling
                    total_loaded = 0
                    
                    for i in range(0, len(df), chunk_size):
                        try:
                            chunk = df.iloc[i:i+chunk_size]
                            for _, row in chunk.iterrows():
                                try:
                                    session.execute(
                                        text(f"""
                                        INSERT INTO {full_table_name} (job_title, embedding)
                                        VALUES (:job_title, :embedding)
                                        """),
                                        {
                                            'job_title': row['jobTitle'],
                                            'embedding': pickle.dumps(row['processed_embedding'])
                                        }
                                    )
                                except Exception as e:
                                    logger.error(f"Error inserting row: {str(e)}")
                                    continue
                            
                            session.commit()
                            total_loaded += len(chunk)
                            logger.info(f"Processed {total_loaded}/{len(df)} rows for {model}")
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {i} for {model}: {str(e)}")
                            session.rollback()
                            continue
                    
                    # Verify the data was loaded correctly
                    loaded_count = session.execute(
                        text(f"SELECT COUNT(*) FROM {full_table_name}")
                    ).scalar()
                    logger.info(f"Total records loaded into {full_table_name}: {loaded_count}")
                    
                    if loaded_count == 0:
                        raise ValueError(f"No data was loaded into {full_table_name}")
                    
                    if loaded_count != len(df):
                        logger.warning(f"Data count mismatch: loaded {loaded_count}, expected {len(df)}")
                    
                    session.commit()
                    logger.info(f"Successfully loaded {model} embeddings")
                    
                except Exception as e:
                    logger.error(f"Error loading embeddings from {file_path}: {str(e)}")
                    session.rollback()
                    continue
                
        except Exception as e:
            logger.error(f"Error in _load_embeddings_to_db: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()
            gc.collect()

    def find_similar_jobs(self, job_title: str, model: str, top_k: int) -> Optional[tuple]:
        """Find similar jobs using the database"""
        session = SessionLocal()
        try:
            table_name = self._get_table_name(model)
            full_table_name = f"{table_name}_embeddings"
            
            # First check if the table exists and has data
            table_exists = session.execute(
                text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = :table_name)"),
                {'table_name': full_table_name}
            ).scalar()
            
            if not table_exists:
                logger.error(f"Table {full_table_name} does not exist")
                return None, None
            
            # Check if the table has any data
            row_count = session.execute(
                text(f"SELECT COUNT(*) FROM {full_table_name}")
            ).scalar()
            
            if row_count == 0:
                logger.error(f"Table {full_table_name} is empty")
                return None, None
            
            logger.info(f"Found {row_count} records in {full_table_name}")
            
            # Get a sample of job titles for debugging
            sample_titles = session.execute(
                text(f"SELECT job_title FROM {full_table_name} LIMIT 5")
            ).fetchall()
            logger.info(f"Sample job titles in database: {[t[0] for t in sample_titles]}")
            
            # Try multiple search strategies
            search_strategies = [
                # Exact match
                (f"job_title = :job_title", {'job_title': job_title}),
                # Case-insensitive exact match
                (f"LOWER(job_title) = LOWER(:job_title)", {'job_title': job_title}),
                # Contains match
                (f"job_title ILIKE :pattern", {'pattern': f'%{job_title}%'}),
                # Word boundary match
                (f"job_title ~* :pattern", {'pattern': f'\\m{job_title}\\M'}),
                # Any word match
                (f"job_title ~* :pattern", {'pattern': f'\\m{job_title}'}),
            ]
            
            result = None
            for strategy, params in search_strategies:
                logger.info(f"Trying search strategy: {strategy}")
                result = session.execute(
                    text(f"""
                    SELECT embedding, job_title
                    FROM {full_table_name}
                    WHERE {strategy}
                    LIMIT 1
                    """),
                    params
                ).first()
                
                if result:
                    logger.info(f"Found match using strategy: {strategy}")
                    logger.info(f"Matched job title: {result[1]}")
                    break
            
            if not result:
                logger.warning(f"No embedding found for job title: {job_title}")
                return None, None
            
            # Load FAISS index
            try:
                index_path = f"data/{table_name}_index.faiss"
                if not os.path.exists(index_path):
                    logger.error(f"FAISS index file not found: {index_path}")
                    return None, None
                
                index = faiss.read_index(index_path)
                
                # Handle the embedding data properly
                embedding_data = result[0]
                if isinstance(embedding_data, memoryview):
                    # If it's a memoryview, convert to bytes
                    embedding_data = bytes(embedding_data)
                elif isinstance(embedding_data, str):
                    # If it's a hex string, convert from hex
                    embedding_data = bytes.fromhex(embedding_data)
                
                query_embedding = pickle.loads(embedding_data).reshape(1, -1)
                
                # Search for similar jobs
                distances, indices = index.search(query_embedding, top_k + 1)
                
                # Get the similar job titles
                similar_jobs = []
                for idx in indices[0][1:]:  # Skip the first result as it's the query itself
                    job = session.execute(
                        text(f"""
                        SELECT job_title
                        FROM {full_table_name}
                        OFFSET :offset
                        LIMIT 1
                        """),
                        {'offset': idx}
                    ).first()
                    if job:
                        similar_jobs.append(job[0])
                
                if similar_jobs:
                    logger.info(f"Found {len(similar_jobs)} similar jobs")
                    logger.info(f"Similar jobs: {similar_jobs}")
                    return similar_jobs, distances[0][1:]
                else:
                    logger.warning("No similar jobs found")
                    return None, None
                    
            except Exception as e:
                logger.error(f"Error in FAISS search for {model}: {str(e)}")
                return None, None
        except Exception as e:
            logger.error(f"Error in find_similar_jobs: {str(e)}")
            return None, None
        finally:
            session.close()

    def compute_metrics(self, job_title: str, similar_jobs: List[str], similarities: np.ndarray, top_k: int):
        """Compute evaluation metrics for the search results"""
        query_keywords = set(job_title.lower().split())
        relevant_jobs = [job for job in similar_jobs if 
                        any(keyword in job.lower() for keyword in query_keywords if len(keyword) > 3)]
        
        relevant_retrieved = len(relevant_jobs[:5])
        precision_at_5 = relevant_retrieved / 5 if len(similar_jobs) >= 5 else 0
        recall_at_5 = relevant_retrieved / len(relevant_jobs) if len(relevant_jobs) > 0 else 0
        
        mrr = 0
        for rank, job in enumerate(similar_jobs, 1):
            if job in relevant_jobs:
                mrr = 1 / rank
                break
        
        true_relevance = [1 if job in relevant_jobs else 0 for job in similar_jobs[:5]]
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

    def run_search(self, job_title: str, top_k: int):
        """Run the search across all embedding models"""
        for model in self.embedding_files.keys():
            similar_jobs, similarities = self.find_similar_jobs(job_title, model, top_k)
            
            if similar_jobs is not None:
                self.results[model]['similar_jobs'] = similar_jobs
                self.results[model]['similarities'] = similarities
                
                metrics = self.compute_metrics(job_title, similar_jobs, similarities, top_k)
                for metric, value in metrics.items():
                    self.results[model][metric] = value

# Define file paths and embedding column names
embedding_files = {
    'GloVe': ('jobs_with_glove_embeddings 2.csv', 'glove_embeddings'),
    'SpaCy': ('jobs_with_spacy_embeddings 2.csv', 'spacy_embeddings'),
    'Sentence-Transformers': ('jobs_with_sentence_embeddings 2.csv', 'embeddings')
}

# Initialize the JobEmbeddingSearch class
try:
    job_search = JobEmbeddingSearch(embedding_files)
    logger.info("JobEmbeddingSearch initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize JobEmbeddingSearch: {str(e)}")
    raise

# Load embeddings into database (only needed once)
if not os.path.exists('data/embeddings_loaded'):
    os.makedirs('data', exist_ok=True)
    try:
        job_search._load_embeddings_to_db()
        with open('data/embeddings_loaded', 'w') as f:
            f.write('1')
        logger.info("Embeddings loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {str(e)}")
        raise

@app.post("/search/")
async def search_job(request: JobSearchRequest):
    try:
        job_title = request.job_title
        top_k = request.top_k
        
        # Perform job search and calculate metrics
        job_search.run_search(job_title, top_k)
        
        # Return results as a dictionary
        return job_search.results
    except Exception as e:
        logger.error(f"Error in search_job endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
