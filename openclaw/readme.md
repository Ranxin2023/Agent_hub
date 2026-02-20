# Open Claw
## Table of Contents
- [What is OpenClaw?](#1-what-is-openclaw)
    - [What Does "Self-Hosted Gateway" Mean?](#what-does-self-hosted-gateway-mean)
    - [What Problem Does It Solve](#what-problem-does-this-solve)
- [Who is it For?](#2-who-is-it-for)
    - [Developers](#developers)
    - [Power Users](#power-users)
- [Features](#features)
    - [Channels](#1-channels)
    - [Plugins](#2-plugins)
    - [Routing](#3-routing)
- [Gateway Architecture](#gateway-architecture)
- [Agent Runtime](#agent-runtime)
    - [Overview](#overview)
    - [Workspace](#workspace-required)
    - [BootStrap](#bootstrap-files-injected)
        - [Agents.md](#1-agentsmd)
        - [SOUL.md](#2-soulmd)
        - [TOOLS.md](#3-toolsmd)
- [Openclaw Skills](#openclaw-skills)
    - [What Are Skills](#what-are-skills)
    - [What's Inside A Skill](#whats-inside-a-skill)
    - [Why Skills Exist](#why-skills-exist)
    - [AgentSkills-Compatible](#agentskills-compatible)
- [Slash](#slash)
## 1. What is OpenClaw?
- *“OpenClaw is a self-hosted gateway that connects your favorite chat apps – WhatsApp, Telegram, Discord, iMessage, and more – to AI coding agents like Pi.”*
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

## Features
### 1. Channels
- *WhatsApp, Telegram, Discord, and iMessage with a single Gateway.*
#### What this really means
- OpenClaw can connect multiple messaging platforms to **one central AI brain**.
- Instead of building:
    - A WhatsApp bot
    - A Telegram bot
    - A Discord bot
    - An iMessage integration
- You connect all of them to the Gateway once.
#### Architecture implication
```markdown
WhatsApp
Telegram
Discord
iMessage
     ↓
   Gateway
     ↓
   AI Agent

```
#### Why this is powerful
- Unified memory across platforms
- No duplicated bot logic
- One AI personality everywhere
- Easier maintenance

### 2. Plugins
- *Add Mattermost and more with extensions.*
#### What are plugins here?
- Plugins are **extensible connectors or capability modules**.
- They can:
    - Add support for new chat platforms
    - Add new tool integrations
    - Add custom logic
    - Extend routing rules
#### Think of it like:
- Browser extensions
- LangChain tools
- Middleware in Express.js
#### Example
- You could write a plugin that:
    - Connects to Jira
    - Connects to GitHub
    - Connects to your CRM
    - Adds Slack support
#### Why it matters
- It prevents the core from becoming bloated.
- 
### 3. Routing
- *Multi-agent routing with isolated sessions.*
- This is one of the most advanced features.
## Gateway Architecture
### 1. “A single long-lived Gateway owns all messaging surfaces”
- **What does “long-lived” mean?**
    - The Gateway is:
        - A persistent process
        - Always running
        - Not serverless
        - Not stateless
        - Not per-request

### 2. Control-plane clients connect via WebSocket
- macOS app, CLI, web UI, automations connect over WebSocket
#### What is the control plane?
- Control plane = management layer.
- Not user chat traffic.
- These are admin / management clients.
- Examples:
    - Web dashboard
    - CLI
    - macOS companion app
    - Automation scripts
- They connect over:
```css
ws://127.0.0.1:18789
```
## Agent Runtime
### Overview
- “OpenClaw runs a single embedded agent runtime derived from pi-mono.”

### Workspace (Required)
- This is one of the most critical design decisions.
    - OpenClaw uses a single agent workspace directory as the agent’s only working directory (cwd) for tools and context.

### Bootstrap files (injected)
- These live inside your workspace directory.
#### 1. AGENTS.md
- Operating instructions + “memory”
- This is like:
    - System rules
    - Long-term instructions
    - Persistent operational memory
- It might contain things like:
```pgsql
You are a coding assistant specialized in Python and TypeScript.
Prefer concise answers.
Always use tools when file access is needed.

```
- This is persistent across sessions.
- It is injected into every new session.
#### 2. SOUL.md
- Persona, boundaries, tone
- This controls personality.
- Example:
```vbnet
Tone: Calm, precise, slightly playful.
Boundaries: Do not give medical advice.
Style: Use bullet points where helpful.

```
- This is separated from AGENTS.md intentionally.
- Why?
    - You might want to change tone without changing logic.
    - Clean separation of behavior vs personality.
#### 3. TOOLS.md
- User-maintained tool notes (e.g. img, sag, conventions)
- This is for tool usage conventions.
- Example:
```css
Use img for image generation.
Use sag for structured agent generation.
Never run shell commands without explaining why.

```
- This gives the agent guidance on how tools should be used.
- This is brilliant because:
    - You can customize tool behavior without modifying code.

#### 4. BOOTSTRAP.md
- *One-time first-run ritual (deleted after completion)*
- This is special.
- It only exists for:
    - A brand new workspace.
- Purpose:
    - Initial setup instructions
    - One-time onboarding
    - Agent self-initialization
- Example:
```pgsql
Ask the user what their preferred language is.
Create a project scaffold.
Initialize config files.

```
- After it's completed:
- You delete it.
#### 5. IDENTITY.md
- *Agent name/vibe/emoji*
- Example:
```makefile
Name: Nova
Emoji: 🌌
Vibe: Futuristic AI research assistant

```
- This allows:
    - Branding
    - Multiple personalities
    - Multi-agent setups
- Very clean separation of identity from logic.
#### 6. USER.md
- *User profile + preferred address*
- Example:
```pgsql
User name: John
Preferred address: Boss
Primary interests: AI, full-stack engineering
```
- This gives the agent user context without storing it in code.

## Openclaw Skills
### What Are Skills?
- “OpenClaw uses AgentSkills-compatible skill folders to teach the agent how to use tools.”
- A skill is:
    - A folder
    - Containing a `SKILL.md` file
    - With YAML frontmatter + instructions
    - That tells the agent how to use a tool
- So skills are not executable code.
- They are structured instructional modules.
### What’s Inside a Skill?
- Each skill is a directory like:
```code
skills/
   web-search/
      SKILL.md
   image-gen/
      SKILL.md
```
- Inside `SKILL.md`:
```markdown
---
name: web-search
description: Use this skill to search the internet.
tools:
  - search
---

When the user asks for recent information,
call the search tool with a concise query.
```
- YAML frontmatter = metadata
- Markdown body = usage instructions

### Why Skills Exist
- Without skills:
    - The agent has tools
    - But doesn’t know when to use them
    - Or how to format arguments properly
- Skills provide:
    - Tool usage rules
    - Behavioral guidelines
    - Conventions
    - Contextual hints
- They are like mini tool manuals.

### AgentSkills-Compatible
- This likely means:
    - OpenClaw follows a shared spec called “AgentSkills.”
- So skills:
    - Follow a standard format
    - Can be reused across systems
    - Are portable
### Locations and Precedence
- Skills are loaded from three places.
#### 1. Bundled Skills (Lowest Priority)
- Shipped with:
    - npm package
    - OpenClaw.app
- These are default skills.
- Example:
    - File editing
    - Shell execution
    - Web search
- They are built-in.
#### 2. Managed / Local Skills
- Location:
```code 
~/.openclaw/skills
```
- These are:
    - User-installed skills
    - Persistent across workspaces
    - Custom tool definitions
- If you override a bundled skill here, this version wins.
#### 3. Workspace Skills (Highest Priority)
```code 
<workspace>/skills
```
- These are:
    - Project-specific
    - Temporary
    - Contextual
    - Highly customized

### Per-agent vs shared skills
#### What Is a Multi-Agent Setup?
- A multi-agent setup means:
- You are running more than one agent, for example:
    - `coding-agent`
    - `research-agent`
    - `assistant-agent`
- Each agent has:
    - Its own workspace
    - Its own memory
    - Its own skills folder (potentially)
- So instead of one AI brain, you have multiple specialized agents.

#### Per-Agent Skills
- “Per-agent skills live in `<workspace>/skills` for that agent only.”
- This means:
    - Each agent has its own workspace directory:
        - Example:
        ```code
        workspace-coding/
        workspace-research/
        workspace-assistant/
        ```
    - Inside each workspace:
        ```code
        workspace-coding/skills/
        workspace-research/skills/
        workspace-assistant/skills/
        ```
    - These skills are:
        - Visible only to that specific agent
        - Not shared with other agents
        - Highest precedence
#### Shared Skills
- *“Shared skills live in `~/.openclaw/skills` and are visible to all agents.”*
- Location:
```code
~/.openclaw/skills
```
- These are:
    - Global skills
    - Machine-wide
    - Available to every agent
- Example:
```code
~/.openclaw/skills/web-search/
~/.openclaw/skills/summarize/
```
### Precedence Rule (Very Important)
- The order is:
1.  `<workspace>/skills`        (highest)
2. ` ~/.openclaw/skills`
3. `bundled skills`            (lowest)
4. `extraDirs (even lower)`

## Plugins + skills
### Big Idea
- *Plugins can ship their own skills.*
- This means:   
    - A plugin is not just executable logic.
    - It can also include:
        - Skill folders
        - `SKILL.md` files
        - Tool usage instructions
- So when you install a plugin, you automatically extend the agent’s intelligence.

### What Is a Plugin?
- A plugin is:
    - An installable extension
    - That can add:
        - Tools
        - Capabilities
        - 
## Slash
### Big Idea
- There are two control systems:
1. **Commands** → standalone `/command`
2. **Directives** → behavior modifiers like `/verbose`
### Commands
- *Commands are standalone messages that start with /*
- Example:
```code 
/help
/status
/whoami
```
- These must be sent as:
    - A single message
    - Starting with `/`
    - Nothing else in the message
- Example (valid): 
```code 
/status
```
### Directives
- Directives are different from commands.
- Examples:
#### How Directives Work
- **1. Inline Directives (Normal Chat)**
```code
/verbose Explain quantum computing.
```
- What happens:
    - `/verbose` is removed before the model sees the message.
    - The remaining message is sent to the model.
    - The directive acts as a hint.
    - It does NOT persist.
- **2. Directive-Only Message**
```code 
/verbose
```