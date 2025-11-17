from openai import OpenAI
import requests
import json
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

# --- Fake Web Search Function (Replace with real API) ---
def search_web(query):
    """
    Example search using NewsAPI or your own backend.
    Replace 'YOUR_API_KEY' with a valid key if needed.
    """
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey=YOUR_API_KEY"
    response = requests.get(url)
    articles = response.json()["articles"][:3]  # take top 3 articles
    return [a["title"] + " - " + a.get("description", "") for a in articles]


# --- The Simple Agent ---
def news_agent(user_query):
    print("User asked:", user_query)

    # Step 1 — Plan
    plan = f"Search the web for: {user_query}"
    print("Agent Plan:", plan)

    # Step 2 — Search the Web (Tool Call)
    raw_results = search_web("AI News")
    print("Raw Web Results:", raw_results)

    # Step 3 — Summarize Results
    prompt = f"""
    Summarize the following AI news into a short clear answer.

    News:
    {json.dumps(raw_results, indent=2)}
    """

    summary = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

    return summary




# ===========================
# Streamlit UI
# ===========================
import streamlit as st

st.title("📰 AI News Agent")
st.write("Ask for any kind of latest news. The agent will search, read, and summarize it!")

# User Input
query = st.text_input("Enter a news topic:", "Latest AI News")

# When button is pressed
if st.button("Search"):
    with st.spinner("Fetching news & summarizing..."):
        answer = news_agent(query)
    
    st.subheader("Agent Answer:")
    st.write(answer)