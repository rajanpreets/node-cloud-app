import streamlit as st
import pinecone
from sentence_transformers import SentenceTransformer
from config import PINECONE_API_KEY, PINECONE_ENVIRONMENT
from paypal import is_subscribed
from groq_summarizer import summarize_results

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index("job-listings")

# Load Sentence Transformer Model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

st.title("AI-Powered Job Search")

# Check subscription
user_email = st.text_input("Enter your email to access premium jobs:")
if not is_subscribed(user_email):
    st.warning("You need a $5/month subscription to access jobs.")
    st.stop()

# Job search input
query = st.text_input("Enter job title, skill, or company:")

if st.button("Search"):
    if query:
        vector = model.encode(query).tolist()
        results = index.query(vector=vector, top_k=5, include_metadata=True)

        if results["matches"]:
            summary = summarize_results(results["matches"])
            st.subheader("Job Search Summary")
            st.write(summary)

            st.subheader("Job Listings")
            for match in results["matches"]:
                st.markdown(f"### {match['metadata']['title']} at {match['metadata']['company']}")
                st.write(f"Location: {match['metadata']['location']}")
                st.write(f"Description: {match['metadata']['description']}")
                st.write(f"Match Score: {match['score']:.2f}")
                st.markdown("---")
        else:
            st.warning("No matching jobs found.")
    else:
        st.warning("Please enter a search query.")
