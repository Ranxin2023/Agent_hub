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
    - [A Single Long Lived Gateway Owns ALl Messaging Surfaces](#1-a-single-long-lived-gateway-owns-all-messaging-surfaces)
    - [Control Plane Clients COnnect Via WebSocket](#2-control-plane-clients-connect-via-websocket)
        - [What Is the Control Plane](#what-is-the-control-plane)
    - [Nodes Also Connect Over WebSocket](#3-nodes-also-connect-over-websocket)
        - [There Are Two Types of WebSocket Clients](#there-are-two-types-of-websocket-clients)
        - [What Are Explicit Caps/Commands](#what-are-explicit-capscommands)
    - [one Gateway Per Host](#4-one-gateway-per-host)
- [Agent Runtime](#agent-runtime)
    - [Overview](#overview)
    - [Workspace](#workspace-required)
    - [BootStrap](#bootstrap-files-injected)
        - [Agents.md](#1-agentsmd)
        - [SOUL.md](#2-soulmd)
        - [TOOLS.md](#3-toolsmd)
- [Canvas](#canvas)
    - [What It Is](#what-it-is)
    - [Key Capabilities](#key-capabilities)
    - [Common Commands](#common-commands)
        - [Openclaw Canvas Push - Send Content to Canvas](#1-openclaw-canvas-push--send-content-to-canvas)
        - [Openclaw Canvas Reset - Clear Canvas](#2-openclaw-canvas-reset--clear-canvas)
- [Openclaw Skills](#openclaw-skills)
    - [What Are Skills](#what-are-skills)
    - [What's Inside A Skill](#whats-inside-a-skill)
    - [Why Skills Exist](#why-skills-exist)
    - [AgentSkills-Compatible](#agentskills-compatible)
- [Slash](#slash)
    - [Big Idea](#big-idea-1)
    - [Commands](#commands)
    - [Directives](#directives)
        - [How Directives Work](#how-directives-work)
        - [Why Stripo Directives](#why-strip-directives)
        - [Authorization Rules](#authorization-rules)
        - [Inline Shortcuts](#inline-shortcuts)
- [ClawHub](#clawhub)
    - [What Is ClawHub](#what-is-clawhub)
    - [What Is Being Shared](#what-is-being-shared)
    - [What You can Do With ClawHub](#what-you-can-do-with-clawhub)
        - [Search Skills](#1-search-skills)
        - [Install Skills](#2-install-skills)
        - [Update Skills](#3-update-skills)
- [Build In Tools](#build-in-tools)
    - [Lobster](#lobster)
        - [What Is Lobster](#what-is-lobster)
        - [What Problem Does It Solve](#what-problem-does-it-solve)
        - [What Lobster Does Instead](#what-lobster-does-instead)
    - [LLM Task](#llm-task)
        - [What is LLM Task](#what-is-llm-task)
        - [Why It Exists](#why-it-exists)
    - [Exec Tools](#exec-tool)
        - [What Is The Exec Tool](#what-is-the-exec-tool)
        - [Foreground vs Background Execution](#foreground-vs-background-execution)
        - [BackGround Sessions are Per Agent](#background-sessions-are-per-agent)
        - [Parameters](#parameters)
        - [Example Use Cases](#example-use-cases)
    - [Web Tools](#web-tools)
        - [What Are Web Tools](#what-are-web-tools)
        - [Web Search](#1-web_search)
        - [Web Fetch](#2-web_fetch)
    - [Apply Patch Tools](#apply_patch-tools)
        - [What Is Apply Patch](#what-is-apply_patch)
        - [Why Not Just Use Edit](#why-not-just-use-edit)
        - [Patch Format Structure](#patch-format-structure)
        - [File Operations Supported](#file-operations-supported)
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
#### Multi-agent routing
- This means the Gateway can decide:
    - Coding question → Coding agent
    - Weather question → Weather agent
    - Math question → Math tool agent
    - Admin command → System agent
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
### 3. Nodes also connect over WebSocket
- *Nodes (macOS/iOS/Android/headless) connect over WebSocket, declaring role: node*
#### There are two types of WebSocket clients:
- **Control clients**
    - Web UI
    - CLI
    - Admin tools
- **Nodes**
    - Devices
    - Mobile apps
    - Headless instances
- When a node connects, it declares:
```code
role: node
```
#### What are explicit caps/commands?
- Caps = capabilities.
- When a node connects, it might declare:
    - I can render Canvas
    - I can capture camera input
    - I can receive push notifications
    - I can run local commands
- The Gateway then knows what this node can do.
- This allows:
    - Distributed AI systems
    - Device-aware routing
    - Capability-based orchestration
### 4. “One Gateway per host”
- This is a strict architectural rule.
- You should NOT run:
    - 2 Gateways on same machine
    - Multiple WhatsApp sessions from same host
- Why?
    - WhatsApp sessions must be unique
    - State must not fork
    - Ports would conflict
    - Session files could corrupt
- This ensures:
    - One authority
    - One state store
    - One routing brain

### 5. The Canvas Host
- The canvas host is served by the Gateway HTTP server under:
```code
The canvas host is served by the Gateway HTTP server under:
```
#### What is Canvas?
- Canvas likely means:
    - A dynamic UI environment
    - Agent-editable HTML/CSS/JS
    - Structured rendering surface
- Instead of plain chat, AI can:
    - Render forms
    - Render forms
    - Render interactive components

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

## Canvas
### What it is
- Canvas in OpenClaw is a **live, agent-driven visual workspace** — essentially a display surface that your AI agent can render content to and control in real time.
- Here's what it does in practice:
    -  Canvas is a separate hosted surface (running on port `18793` by default) that acts like a screen your agent can "push" content to — think of it as a live visual output panel alongside your chat.
### Key capabilities:
- The agent can push content, UI, or code output directly to the Canvas
- You can view it as a live, updating display
- iOS and Android nodes expose the Canvas surface on mobile devices
- The macOS app includes voice trigger forwarding and the Canvas surface together
### Common Commands
#### 1. `openclaw canvas push` — Send Content to Canvas
- This is the primary command for rendering content on the Canvas surface. It uses OpenClaw's A2UI (Agent-to-UI) system to push structured UI components.
- **What it does**
    - Sends A2UI-formatted JSONL data to the Canvas panel, which renders it as a live visual interface — text, columns, buttons, etc.
- **Example JSONL payload:**
```jsonl
{
    "surfaceUpdate":
    {
        "surfaceId":"main",
        "components":[
            {
                "id":"root",
                "component":
                {
                    "Column":{"children":{"explicitList":["title","content"]}}
                }
            },
            {
                "id":"title",
                "component":{"Text":{"text":{"literalString":"My Dashboard"},"usageHint":"h1"}}
            },
            {
                "id":"content",
                "component":{"Text":{"text":{"literalString":"Agent report goes here."},"usageHint":"body"}}
            }
        ]
    }
}
{
    "beginRendering":{"surfaceId":"main","root":"root"}
}
```
#### 2. `openclaw canvas reset` — Clear Canvas
- **What it does**: Wipes the Canvas surface clean, removing whatever content was previously pushed. Returns it to the blank scaffold (or `index.html` if one exists).
- 
### How Canvas is used:
- Instead of only reading text replies in WhatsApp/Telegram, the agent can simultaneously render something visual on Canvas
- Possible visual outputs include: a dashboard, a rendered webpage summary, an image, or dynamic UI via OpenClaw's A2UI system
#### In the context of your project files (`web_socket_agent.py`, `server.py`, etc.):
- Your agents are currently only doing text-based chat over WebSocket/HTTP
- To use Canvas, you would need to send canvas-specific event types through the OpenClaw Gateway WebSocket protocol
- This means replacing or supplementing plain `send_message` events with canvas-targeted commands

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
- Example (invalid):
```code
/status what is going on?
```
- Commands are processed immediately by the Gateway.
- The model never sees them.
### Directives
- Directives are different from commands.
- Examples:
```code
/think
/verbose
/reasoning
/elevated
/exec
/model
/queue
```
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
- When the message contains only directives:
    - It persists to the session.
    - It modifies session-level behavior.
    - Gateway replies with acknowledgment.
- Now all future messages may be verbose.
#### Why Strip Directives?
- The model should not see:
```code
/verbose Explain something
```
- Instead, it sees:
```code
Explain something
```
- The Gateway handles the behavior flag separately.
- This keeps:
    - Clean prompts
    - Clear control boundaries
    - Predictable behavior
#### Authorization Rules
- Directives only work for:
    - *Authorized senders.*
- If not authorized:
    - Directives are treated as plain text.
    - No special behavior happens.
- Authorization is determined by:
    - 1. `commands.allowFrom`
    - 2. Channel allowlists / pairing
    - 3. `commands.useAccessGroups`
- If `commands.allowFrom` is set:
    - It overrides everything else.
    - Only those users can issue directives.
- This prevents random users from changing system behavior.
#### Inline Shortcuts
- These are special commands:
```code
/help
/commands
/status
/whoami
/id
```
- What does the above special commands do
1. `/help`
- **What it does**
    - Displays basic usage help.
    - Typically shows:
        - How slash commands work
        - Available directives
        - How to use `/bash`
        - How authorization works
        - Possibly current channel info
- **What it does NOT do**
    - It does not call the LLM
    - It does not generate AI content
    - It does not affect session state
2. `/commands`
- **What it does**
    - Lists all available slash commands and directives.
    - For example, it might show:
    ```
    /help
    /status
    /whoami
    /verbose
    /model
    /bash
    ...
    ```
- This is useful for:
    - Discovering features
    - Debugging permissions
    - Seeing which directives are enabled
- Again:
    - Handled by Gateway
    - No LLM involvement

3. `/status`
- **What it does**
    - Shows the current runtime state of the Gateway.
    - It may display things like:
        - Current model in use
        - Session ID
        - Active directives
        - Connected channels
        - Node status
        - Plugin status
        - Workspace path
    - This is basically:
        - *“System health + configuration snapshot”*
    - Very useful for debugging.
4. `/whoami`
- **What it does**
    - Tells you who the Gateway thinks you are.
    - This may include:
        - Your internal user ID
        - Channel identity
        - Authorization group
        - Whether you are authorized for directives
        - Pairing status
    - This is important because:
        - Slash directives only work for authorized users.
    - So `/whoami` helps confirm:
        - Why something worked
        - Why something didn’t

5. `/id`
- **What it does**
    - Shows your sender ID or channel-specific identifier.
- **For example:**
    - WhatsApp JID
    - Telegram user ID
    - Discord user ID
- This is useful for:
    - Setting up `commands.allowFrom`
    - Debugging permissions
    - Pairing devices
    - Granting admin rights

### Key Difference Between Commands and Directives

| **Type**  | **Format** |**Who Handles**|**Visible to Model?**| **Purpose**   |
| --------- | ---------- | ----------- | ----------------- | ------------- |
| Command   | `/status`  | Gateway     | ❌ No              | System action |
| Directive | `/verbose` | Gateway     | Sometimes         | Behavior hint |

## ClawHub
### What Is ClawHub?
- *“ClawHub is the public skill registry for OpenClaw.”*
- ClawHub is:
    - A central website/service
    - Where skills are published
    - Shared publicly
    - Versioned
    - Discoverable
### What Is Being Shared?
- Skills.
### What You Can Do With ClawHub
#### 1. Search skills
#### 2. Install skills
- CLI might do something like:
```code
openclaw skills install github-issues
```
- This likely installs into:
```code
~/.openclaw/skills
```
#### 3. Update skills
```code 
openclaw skills update github-issues
```
#### 4. Publish skills
- If you create your own skill:
```code
my-skill/
   SKILL.md
```
- You can publish it to ClawHub.
### What ClawHub Is (Architecturally)
#### 1. A Public Registry
- Like:
    - npmjs.com
    - PyPI
    - Docker Hub
    - VSCode Marketplace
- But specifically for OpenClaw skills.
#### 2. A Versioned Store
- Skills are versioned.
- Example:
```code
github-skill@1.0.0
github-skill@1.1.0
github-skill@2.0.0
```
- You can lock versions
- Upgrade safely
- Avoid breaking changes
#### 3. A Discovery Surface
- ClawHub supports:
    - Search
    - Tags

## Build-In Tools
### Lobster
#### What Is Lobster?
- *Typed workflow runtime for OpenClaw — composable pipelines with approval gates.*
- In simple terms:
    - Lobster is a **workflow engine** inside OpenClaw.

#### What Problem Does It Solve?
- Without Lobster:
    - The LLM must:
    1. Think
    2. Call tool
    3. Call tool
    4. Think again
    5. Call next tool
    6. Call next tool
- Each step:
    - Costs tokens
    - Requires reasoning
    - Is probabilistic
- This creates:
    - Expensive orchestration
    - Unpredictable behavior
    - Fragile multi-step flows
#### What Lobster Does Instead
```code
Pipeline:
  Step 1 → Fetch data
  Step 2 → Transform data
  Step 3 → Send email (approval required)
  Step 4 → Log result
```
- Then OpenClaw runs it as:
    - *One deterministic operation.*
- The LLM calls Lobster once.
- Lobster handles everything internally.

### LLM-Task
#### What Is `llm-task`?
- *“llm-task is an optional plugin tool that runs a JSON-only LLM task and returns structured output (optionally validated against JSON Schema).”*
- In simple terms:
    - llm-task is a controlled way to call an LLM and force it to return strict JSON output instead of free-form text.
- It is:
    - A tool
    - Not raw chat
    - Not free reasoning
    - Not free reasoning
#### Why It Exists
- Normally, if you ask an LLM:
    - Normally, if you ask an LLM:
- It might return:
    ```code
    Name: John
    Email: john@example.com
    ```
- But maybe not in consistent format.
- That’s bad for workflows.
- Instead, `llm-task` enforces:
```json
{
  "name": "John",
  "email": "john@example.com"
}
```
- And optionally validates it against a schema.
### Exec Tool
#### What Is the Exec Tool?
- *“Run shell commands in the workspace.”*
- The agent can execute commands like:
```bash
ls
npm install
python script.py
git status
```
- But only:
    - Inside the workspace (by default)
    - Under controlled security rules
    - With optional approvals
    - With optional sandboxing
#### Foreground vs Background Execution
- **1. Foreground (default)**
    - The command runs:
        - The command runs:
        - Waits for completion
        - Returns output
- **2. Background (via `process`)**
    - If background execution is allowed:
        - The command runs asynchronously
        - The agent gets a process handle
        - It can resume or check later
- What If process Is Disallowed?
    - It ignores `yieldMs`
    - Ignores `background`
    - Runs synchronously

#### **Background Sessions Are Per-Agent**
- If two agents run background jobs:
    - They do NOT see each other’s processes.
    - Process isolation is enforced per agent.
- This prevents:
    - Cross-agent interference
    - Accidental process control
#### Parameters
- `command` (required)
```json
{ "command": "npm run build" }
```
- `workdir`
- `env`
- `yieldMs` (default 10000)
#### Example Use Cases
1. **Run tests**
```json
{
  "command": "npm test"
}
```
2. **Start dev server**
```json
{
  "command": "make deploy",
  "ask": "always"
}
```
3. **Compile code with approval**
```json
{
  "command": "make deploy",
  "ask": "always"
}
```
4. **Run on mobile node**
```json
{
  "command": "ls",
  "host": "node",
  "node": "iphone-01"
}
```

### Web Tools
#### What Are Web Tools?
- OpenClaw includes two built-in web tools:
1. `web_search`
2. `web_fetch`
- They are:
    - Lightweight
    - Non-browser
    - Deterministic
    - Structured
    - Safe by design
- They are not full browser automation.
#### 1. `web_search`
- *Search the web via Brave Search API (default) or Perplexity Sonar.*
- This tool performs web search queries.
- This tool performs web search queries.
- It does NOT scrape pages itself.
#### 2. `web_fetch`
- *HTTP fetch + readable extraction (HTML → markdown/text)*
- This tool:
    - Fetches a URL via HTTP GET
    - Extracts readable content
    - Converts HTML into markdown or plain text
- It does NOT:
    - Execute JavaScript
    - Render dynamic sites
    - Render dynamic sites
#### What It’s Good For
- Static blog posts
- Documentation
- News articles
- Public pages
- README files
#### What It Cannot Do
- Load React apps
- Execute JS-heavy pages
- Handle login sessions
- Fill forms
- Click buttons
#### Important Distinction
| Tool         | Purpose         | Executes JS? |
| ------------ | --------------- | ------------ |
| web_search   | Find URLs       | ❌ No         |
| web_fetch    | Read content    | ❌ No         |
| Browser tool | Full automation | ✅ Yes        |
- Web tools are lightweight and cheap.
- Browser tool is heavy and powerful.

### `apply_patch` Tools
#### What is `apply_patch`
- *“Apply file changes using a structured patch format.”*
- It allows the agent to:
    - Add files
    - Update files
    - Delete files
    - Modify multiple files at once
    - Modify multiple sections inside a file
- Using a **strict patch format**.
#### Why Not Just Use Edit?
- *“Ideal for multi-file or multi-hunk edits where a single edit call would be brittle.”*
- If you use a simple edit tool:
    - It might replace the entire file
    - It might break formatting
    - It may overwrite unintended content
    - It cannot safely modify multiple sections
- apply_patch solves this by:
    - Being diff-based
    - Being explicit
    - Being structured
    - Being predictable
#### Patch Format Structure
- It must follow this format:
```plan text
*** Begin Patch
...
*** End Patch
```
#### File Operations Supported
- **1. Add File**
    ```plain text
    *** Add File: path/to/file.txt
    +line 1
    +line 2
    ```
    - Creates a new file
    - Adds the listed lines
    - Each line starts with `+`
    - The `+` means:
        - *This line is added.*
- **2. Update File**
    ```plain text
    *** Update File: src/app.ts
    @@
    -old line
    +new line
    ```
    - This uses diff-style syntax.
        - `-` means remove
        - `+` means add
        - `@@` marks a hunk (section of changes)
- **3. Delete File**
    ```plain text
    *** Delete File: obsolete.txt
    ```
    - Deletes the file entirely.

#### Multi-File Example
```plain text
*** Begin Patch
*** Add File: new.js
+console.log("Hello");

*** Update File: app.js
@@
-console.log("Old");
+console.log("New");

*** Delete File: old.js
*** End Patch
```
#### Why This Is Important for AI Agents
- Without structured patching:
    - Rewrite entire files
    - Accidentally remove content
    - Break indentation
    - Overwrite unrelated sections
    - Create merge conflicts
- With `apply_patch`:
    - Only specified lines change
    - Changes are deterministic
    - Easier to audit
    - Easier to validate
    - Safer to review