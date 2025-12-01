# Code Explanation of `ReAct.ipynb`
## What ReAct.ipynb Actually Is
### 1. Import & Environment Setup
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
- **Streamlit** → Build the UI for your AI News Agent. Lets users enter text, click buttons, see results interactively.
- **OpenAI()** → The new OpenAI SDK (2025). Used for calling GPT-4.1 in this notebook.
- **requests** → Makes HTTP requests to NewsAPI.
- **json** → Formats JSON safely for the prompt.
- **os** → Access environment variables (API keys).
- **dotenv** → Load `.env` file into environment.
### 2. SAFE SEARCH FUNCTION
- Step 1 — Check API key exists
    - If missing → return readable error.

- Step 2 — Build URL to query NewsAPI
```python
url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={API_KEY}"

```
- Step 3 — Request data
```python
response = requests.get(url)
data = response.json()

```
- 