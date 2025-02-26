import pinecone
from sentence_transformers import SentenceTransformer
import json
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index("job-listings")

# Load Sentence Transformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def vectorize_jobs(job_data):
    """
    Vectorizes job descriptions and stores them in Pinecone.
    """
    vectors = []
    for job in job_data:
        vector = model.encode(job["description"]).tolist()
        vectors.append((job["id"], vector, job))  # Store metadata

    index.upsert(vectors)

# Load job data from JSON
with open("jobs.json", "r") as f:
    jobs = json.load(f)

vectorize_jobs(jobs)
print("Jobs stored successfully in Pinecone!")

