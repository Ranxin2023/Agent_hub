from openai import OpenAI
import requests
from bs4 import BeautifulSoup
# import os
from dotenv import load_dotenv
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR/".env")
client = OpenAI()

# -----------------------------------
# 1️⃣ Scraping Function
# -----------------------------------
def scrape_website(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get all paragraph text
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:4000]  # limit for token safety

    except Exception as e:
        return f"Error scraping website: {str(e)}"


# -----------------------------------
# 2️⃣ WhatsApp Message Handler
# -----------------------------------
def handle_message(message_text):
    """
    This function is called by OpenClaw
    whenever WhatsApp receives a message.
    """

    # If user sends a URL
    if message_text.startswith("http"):
        scraped_text = scrape_website(message_text)

        # Send scraped content to GPT for summary
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "Summarize the webpage clearly."},
                {"role": "user", "content": scraped_text}
            ]
        )

        return response.choices[0].message.content

    # Otherwise normal chat
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are a helpful WhatsApp AI assistant."},
            {"role": "user", "content": message_text}
        ]
    )

    return response.choices[0].message.content
