import websocket
import json
from openai import OpenAI
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

load_dotenv()

client = OpenAI()

GATEWAY_URL = "ws://localhost:18789"  # default gateway \

# ----------------------------
# Scraping Function
# ----------------------------
def scrape_website(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Get all paragraph text
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:4000]  # limit for token safety
    except Exception as e:
        return f"Error scraping website: {str(e)}"


# ----------------------------
# your Agent logic
# ----------------------------
def handle_message(message_text):

    if message_text.lower() == "hello":
        return "Hello from my custom agent!"

    # If user sends a URL
    if message_text.startswith("http://") or message_text.startswith("https://"):
        scraped_text = scrape_website(message_text)

        # Send scraped content to GPT for summary
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the webpage clearly."},
                {"role": "user", "content": scraped_text}
            ]
        )

        return response.choices[0].message.content

    # call GPT by default
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fixed: use valid model name
        messages=[
            {"role": "system", "content": "You are a helpful WhatsApp AI assistant."},
            {"role": "user", "content": message_text}
        ]
    )

    return response.choices[0].message.content


# ----------------------------
# WebSocket callback
# ----------------------------
def on_message(ws, message):
    print("Received:", message)

    data = json.loads(message)

    # 🔥 handle challenge
    if data.get("event") == "connect.challenge":
        nonce = data["payload"]["nonce"]

        ws.send(json.dumps({
            "type": "connect.authenticate",
            "payload": {
                "nonce": nonce,
                "auth": {}
            }
        }))

        print("Sent authentication response")
        return

    # handle normal message
    if "text" in data:
        user_text = data["text"]
        reply = handle_message(user_text)

        ws.send(json.dumps({
            "type": "send_message",
            "text": reply
        }))


def on_error(ws, error):
    print("Error:", error)


def on_close(ws, close_status_code, close_msg):
    print("Closed connection")


def on_open(ws):
    print("Connected to OpenClaw Gateway")


# ----------------------------
# automatically connect
# ----------------------------
def run():
    ws = websocket.WebSocketApp(
        GATEWAY_URL,
        subprotocols=["openclaw"],
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws.run_forever()


if __name__ == "__main__":
    run()
