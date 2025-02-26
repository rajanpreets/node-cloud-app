import requests
from config import GROQ_API_KEY

def summarize_results(job_list):
    """
    Uses Groq API to summarize job search results.
    """
    if not job_list:
        return "No jobs found."

    descriptions = "\n\n".join([f"Job {i+1}: {job['metadata']['title']} at {job['metadata']['company']}. {job['metadata']['description']}" for i, job in enumerate(job_list)])

    prompt = f"""
    Here are job listings found based on a search query:
    {descriptions}

    Summarize these jobs in a brief paragraph highlighting key insights.
    """

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mixtral-8x7b",
        "prompt": prompt,
        "max_tokens": 200
    }

    response = requests.post("https://api.groq.com/v1/chat/completions", headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return "Error in summarizing results."
