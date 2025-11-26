# Full Code Explanation of `ai_news_agent.py`
## 1. Imports & Setup
```python
import streamlit as st
from openai import OpenAI
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

```
- `streamlit`
    - Used to create a simple UI for user input & displaying results.
- `openai`
    - Used to call GPT-4.1 through the new OpenAI Python SDK.
- `requests`
    - 
## 2. SAFE SEARCH FUNCTION (Tool Use)
```python
def search_web(query):
    API_KEY = os.environ.get("NEWSAPI_KEY")

```
- Loads your NewsAPI key from environment variables
### Safety: Missing key
```python
if not API_KEY:
    return ["❌ ERROR: Missing NewsAPI key. Set NEWSAPI_KEY environment variable."]

```
### Building the NewsAPI endpoint
```python
url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={API_KEY}"
response = requests.get(url)

```
- Searches the web for **any topic** (`query`)
- Limits results to English
- Uses your API key

## 3. AI Agent Logic (Reasoning + Summarization)
```python
def news_agent(user_query):
    raw_results = search_web(user_query)

```
### Step 1 — Web Search
- Results from the tool are stored in `raw_results`.
#### Handling errors from the tool
```python
if len(raw_results) == 1 and raw_results[0].startswith("❌"):
    return raw_results[0]

```
### Step 2 — Build prompt for GPT
```python
prompt = f"""
Summarize the following news results into a short, clear explanation.

Results:
{json.dumps(raw_results, indent=2)}
"""

```

### Step 3 — LLM call
```python
summary = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": prompt}]
).choices[0].message.content

```
- GPT reads:
    - the list of article titles
    - the summaries
    - produces a one-paragraph explanation
