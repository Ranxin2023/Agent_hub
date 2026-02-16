# Open Claw
## Table of Contents
- [What is OpenClaw?](#1-what-is-openclaw)
- [Who is it For?](#2-who-is-it-for)
    - [Developers](#developers)
## 1. What is OpenClaw?
- “OpenClaw is a self-hosted gateway that connects your favorite chat apps – WhatsApp, Telegram, Discord, iMessage, and more – to AI coding agents like Pi.”
### What does “self-hosted gateway” mean?
- Self-hosted = You run it on your own computer or server
- Gateway = It acts as a bridge between:
    - Your messaging apps
    - Your AI assistant
- So instead of using a company’s hosted chatbot platform, you run your own **AI bridge server**.
- Think of it like this:
`scss
WhatsApp
Telegram
Discord
     ↓
   OpenClaw (running on your machine)
     ↓
   AI Agent (like Pi, GPT, Claude, etc.)

`scss
### What problem does this solve?
- Normally:
    - AI assistants live inside specific apps
    - You’re dependent on third-party hosting
    - You don’t control your data
- With OpenClaw:
    - You message your AI from any supported chat app
    - The processing happens through your own gateway
    - You stay in control

## 2. Who is it for?
- Developers and power users who want a personal AI assistant they can message from anywhere — without giving up control of their data or relying on a hosted service.
### Developers
- People who:
    - Build AI agents
    - Want custom workflows
    - Use coding agents with tools
### Power users
- People who:
    - Want automation
    - Want their own AI assistant across platforms
    - Care about privacy & control
### Key Idea: Data Control
- Because it’s self-hosted:
    - Your chat messages are not stored by OpenClaw servers
    - You control logging, storage, routing
    - You choose which AI model to connect
## 3. What Makes It Different?
### 3.1 . Self-hosted
- Runs on your hardware, your rules
- Meaning:
    - You deploy it locally or on your server (AWS, DigitalOcean, etc.)
    - You configure API keys
    - You decide data retention
    - No vendor lock-in
### 3.2 Multi-channel
- Instead of building separate bots for:
    - WhatsApp
    - Telegram
    - Discord
- You deploy **one gateway**, and it connects to all of them.
- That means:
    - One AI brain
    - Many chat front-ends
- Very efficient architecture.
### 3.3 Agent-native
- Built for coding agents with tool use, sessions, memory, and multi-agent routing
- This is important.
- It’s not just a chatbot connector.
- It supports:
    - 🧠 Memory
    - 🔧 Tool usage
    - 🔁 Multi-agent workflows
    - 💬 Session management
- This is similar to how your ReAct + LangGraph agent:
    - Uses tools
    - Stores state
    - Manages tool calls
    - Iteratively reasons 
- OpenClaw is designed to support that kind of intelligent, tool-using AI system — not just simple Q&A bots.

### 3.4. Open Source (MIT licensed)
- This means:
    - You can modify it
    - You can fork it
    - You can use it commercially
    - The community contributes improvements
- MIT license = very permissive.
### 3.5 What Do You Need?
- Node 22+, an API key (Anthropic recommended), and 5 minutes.
#### Node 22+
- It’s built using modern JavaScript (Node.js runtime).
#### API key
- Anthropic (Claude) recommended
- Or any supported AI provider
#### 5 minutes
- Install dependencies
- Add API key
- Run server
- Connect chat apps

## Workflow Explanation
![Openclaw Workflow](../images/openclaw_workflow.png)
### Step 1: Chat apps + plugins → Gateway
#### What are “Chat apps + plugins”?
- These are input channels:
    - WhatsApp
    - Telegram
    - Discord
    - iMessage
    - Slack
    - Custom plugins
- They are **just interfaces**.
- They do NOT:
    - Store AI memory
    - Decide routing
    - Handle logic
- They only:
    - Receive user messages
    - Send them to the Gateway
- Think of them as front doors.
### Step 2: The Gateway (Core of the System)
- This is the most important component.
- The Gateway:
    - Receives messages from chat apps
    - Maintains session state
    - Decides which agent handles the message
    - Manages tool usage
    - Stores conversation memory
    - Routes responses back to the correct channel
- It is the **central coordinator**.

### Step 3: Gateway → Output / Consumers
#### 1. Pi Agent
- This is the AI agent itself.
- The Gateway:
    - Sends messages to the agent
    - Passes session context
    - Handles tool calls
    - Receives responses
- This agent could:
    - Use tools
    - Run code
    - Search web
    - Execute workflows
- Think of this like your ReAct-style agent — where the model:
    - Thinks
    - Calls tools
    - Responds
- The Gateway is what orchestrates that loop.

#### 2. CLI
- Command Line Interface.
- This allows you to:
    - Talk to your AI from terminal
    - Debug
    - Send commands
    - Inspect sessions
- Example:
```nginx
openclaw ask "summarize today's messages"
```
#### 3. Web Control UI
- A browser dashboard.
- Used for:
    - Viewing sessions
    - Managing agents
    - Monitoring logs
    - Debugging tool calls
    - Changing settings
- Important:
    - It does NOT store the state.
    - It only reads from the Gateway.