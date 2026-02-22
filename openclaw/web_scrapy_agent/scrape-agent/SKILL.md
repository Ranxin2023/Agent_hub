---
name: scrape-agent
description: Scrape a webpage and summarize its content using AI. Use this skill when the user sends a URL or asks to summarize a webpage.
---

# Web Scraper & AI Summary Skill

## Overview
This skill scrapes a webpage and returns an AI-powered summary using GPT.
Trigger this skill when the user sends a URL or asks to summarize a webpage.

## Trigger Conditions
- User sends a message that starts with `http://` or `https://`
- User says "scrape this", "summarize this page", "总结这个网页", "帮我看看这个链接"

## How to Use This Skill

When triggered, run the scrape agent script using Python:

```bash
python "{baseDir}\scrape_agent.py" "<URL>"
```

Replace `<URL>` with the actual URL from the user's message.

## Expected Output
The script will return a plain-text summary of the webpage. 
Forward this summary directly to the user as your reply.

## Error Handling
- If the script returns an error starting with "Error", tell the user: "Sorry, I couldn't scrape that page. It may be behind a login or blocked bots."
- If the URL is unreachable, suggest the user check if the link is public.

## Notes
- Only works with public URLs (no login required)
- Summarizes up to 4000 characters of page content
- Uses GPT-4.1 for summarization
