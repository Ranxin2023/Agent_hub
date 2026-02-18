from fastapi import FastAPI, Request
from openai import OpenAI
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

app = FastAPI()

# -------------------------
# your Scrape code
# -------------------------
def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:4000]
    except Exception as e:
        return f"Error scraping: {str(e)}"


# -------------------------
# handle message
# -------------------------
def handle_message(message_text):

    if message_text.lower() == "hello":
        return "Hello from my custom scrape agent!"

    if message_text.startswith("http"):
        content = scrape_website(message_text)

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Summarize the webpage clearly."},
                {"role": "user", "content": content}
            ]
        )

        return response.choices[0].message.content

    return "Send me a URL to scrape."


# -------------------------
# Webhook Endpoint
# -------------------------
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()

    # 模拟 WhatsApp message 结构
    message_text = data.get("message")

    reply = handle_message(message_text)

    return {"reply": reply}
