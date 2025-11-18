import streamlit as st
from openai import OpenAI
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

# ============================
# SAFE SEARCH FUNCTION
# ============================
def search_web(query):
    API_KEY = os.environ.get("NEWSAPI_KEY")  # optional env var

    if not API_KEY:
        return ["❌ ERROR: Missing NewsAPI key. Set NEWSAPI_KEY environment variable."]

    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={API_KEY}"
    response = requests.get(url)

    try:
        data = response.json()
    except Exception:
        return ["❌ ERROR: Failed to parse NewsAPI response."]

    # Handle API error responses safely
    if "status" in data and data["status"] == "error":
        return [f"❌ NewsAPI Error: {data.get('message', 'Unknown error')}"]

    if "articles" not in data:
        return ["❌ ERROR: No 'articles' field in NewsAPI response."]

    # Take top 3 articles safely
    articles = data["articles"][:3]

    if not articles:
        return ["⚠ No articles found."]

    return [
        (a.get("title", "No title") + " - " + a.get("description", ""))
        for a in articles
    ]


# ============================
# AI Agent
# ============================
def news_agent(user_query):
    raw_results = search_web(user_query)

    # If search result contains error messages, return them immediately
    if len(raw_results) == 1 and raw_results[0].startswith("❌"):
        return raw_results[0]

    prompt = f"""
    Summarize the following news results into a short, clear explanation.

    Results:
    {json.dumps(raw_results, indent=2)}
    """

    summary = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    return summary


# ============================
# STREAMLIT APP
# ============================
st.title("📰 AI News Agent")
st.write("Ask for news. The agent will search, summarize, and answer.")

query = st.text_input("Enter a topic:", "Latest AI News")

if st.button("Search"):
    with st.spinner("Fetching & summarizing..."):
        answer = news_agent(query)

    st.subheader("Answer:")
    st.write(answer)
