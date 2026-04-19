# Agentic AI Frameworks
## Table of Contents
- [Key Frameworks](#1-key-frameowkrs)
    - [Langgraph](#11-langgraph)
    - [CrewAI](#12-crewai)
    - [Autogen](#13-autogen-ag2)
- [What you Will Learn](#what-you-will-learn)
    - [Identify the core Traits of Agentic AI Systems](#1-identify-the-core-traits-of-agentic-ai-systems)
    - [Describe the Benefits and CHallenges of Multi-Agent Architectures](#2-describe-the-benefits-and-challenges-of-multi-agent-architectures)
- [Key Characteristics of AI Agents](#key-characteristics-of-ai-agents)
## 1. Key Frameowkrs
### 1.1 Langgraph
- Focus: Workflow-based AI systems
### 1.2 CrewAI
- Focus: Multi-agent collaboration
- You create multiple agents with roles like:
    - Researcher
    - Writer
    - Reviewer

- Example:
    - Agent 1 gathers data
    - Agent 2 summarizes
    - Agent 3 validates output
- This mimics a **team of AI workers**

### 1.3 AutoGen (AG2)
- Focus: **Conversation-driven agents**
- Agents talk to each other dynamically
- Example:

## What you will learn
### 1. Identify the core traits of agentic AI systems
- Key traits you’ll learn:
**1. Autonomy**
- The AI can act without constant human input
- It decides next steps based on goals
**2. Goal-driven behavior**
- Agents work toward an objective
**3. Tool usage**
- Agents can call APIs, databases, functions
**4. Memory & state**
- Agents remember past steps or conversations
- Example:
    - Chat history
    - Intermediate reasoning steps

### 2. Describe the benefits and challenges of multi-agent architectures
- Now we move from **one agent → multiple agents working together**
### 3. Compare strengths and use cases of popular open-source frameworks
- This is where the course becomes practical and engineering-focused
- You’ll learn how frameworks differ:
    - **LangGraph**
        - **Best for: structured workflows**
            - Graph-based execution
            - Deterministic + controllable
            - Strong state management
    - **CrewAI**
        - **Best for: role-based agents**
            - Human-like collaboration
            - Easy to assign roles
        - Use when:
            - Tasks can be split into roles
            - You want readable architecture
    - **AutoGen (AG2)**
        - **Best for: conversational agents**
            - Agents talk dynamically
            - Flexible but less controlled
        - Use when:
            - Iterative reasoning is needed
            - Open-ended tasks
## Key Characteristics of AI agents
### 1. Multi-step reasoning
- `“Breaks down complex tasks”`
#### What it means
- An AI agent doesn’t just answer in one shot. It:
    - Decomposes a problem into steps
    - Solves each step sequentially (or iteratively)

- Example
    - User asks:
### 2. Goal-oriented behavior
- *“Ensures decisions and actions achieve an objective”*
- **🔍 What it means**
    - Agents don’t just respond—they optimize toward a goal
- **💡 Example**
    - Goal:
        - *“Get the best answer”*
    - Agent decisions:
        - Should I search?
        - Should I refine?
        - Should I ask clarification?
- **⚖️ Key difference**
| **Normal AI**| **Agent**   |
| ------------ | ----------- |
| Responds     | Plans       |
| Static       | Adaptive    |
| No objective | Goal-driven |

### 3. Decision-making
- *“Chooses the best course of action”*
- **What it means**
    - Use a tool?
    - Ask user?
    - Continue reasoning?
    - Stop?
