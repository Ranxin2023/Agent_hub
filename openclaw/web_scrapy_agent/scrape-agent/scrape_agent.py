import sys
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load .env from parent directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

client = OpenAI()


def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])
        return text[:4000]
    except Exception as e:
        return f"Error scraping website: {str(e)}"


def summarize(url):
    scraped_text = scrape_website(url)

    if scraped_text.startswith("Error"):
        print(scraped_text)
        return

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Summarize the webpage clearly and concisely. If the content is in Chinese, reply in Chinese."},
            {"role": "user", "content": scraped_text}
        ]
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide a URL as argument.")
        print("Usage: python scrape_agent.py <URL>")
        sys.exit(1)

    url = sys.argv[1]
    summarize(url)
