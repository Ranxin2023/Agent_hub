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
- [Summary of CrewAI](#summary-of-crewai)
    - [](#all-tips)
- [Summary of BeeAI](#summary-of-beeai)
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

## Summary of CrewAI
### All tips of CrewAI

- CrewAI is designed for multi-agent collaboration, with agents assigned clear roles and tasks to simulate human-like teamwork

- Tools are standard components in AI workflows (for example, APIs and search engines) that can be used by either the Agent or the Task.

- The Crew object combines agents, tasks, the LLM, and tools into a coordinated workflow.

- CrewOutput captures the final result, task outputs, and token usage, giving a full snapshot of what was generated and its cost

- CrewAI lets you build multi-agent workflows by defining agents with specific roles, goals, and tasks, then grouping them in a Crew for sequential execution

- YAML allows you to define agents and tasks outside of Python, simplifying updates without touching code

- The @CrewBase decorator loads YAML-defined components as methods, making them easy to call and integrate into a Python script or notebook

- Custom functions enhance CrewAI by enabling domain-specific tools that improve flexibility and control

- An agent-centric workflow assigns tools directly to the agent, letting them choose the best tool based on the query

- A task-centric workflow attaches tools to individual tasks, guiding the agent step by step through a fixed process

### 1. What CrewAI is REALLY about
- **CrewAI = a framework for simulating a team of AI agents working together**
### 🤝 2. Multi-Agent Collaboration (Core Idea)
- *“Agents assigned clear roles and tasks”*
#### Example
| Agent      | Role           | Task             |
| ---------- | -------------- | ---------------- |
| Researcher | Finds info     | Search data      |
| Analyst    | Processes info | Extract insights |
| Writer     | Communicates   | Generate report  |

## Summary of BEEAI
### All tips
- BeeAI is a cutting-edge, open-source platform for building production-ready AI agents

- You can create an AI conversation in BeeAI by importing the necessary modules, initializing the chat model, defining the conversation messages, and running the model asynchronously with Python's async and await syntax

- Dynamic prompt templates enable the creation of reusable prompts with variable data

- BeeAI can generate structured outputs using Pydantic schemas

- Conversational memory is managed with the UnconstrainedMemory class

- The key benefits of using the BeeAI framework include modularity, structured outputs, async execution, multi-agent support, standards compliance, and observability

- Agents maintain a persistent state, use external tools, and follow behavioral requirements

- The RequirementAgent class is used to build intelligent, controllable agents

- The UnconstrainedMemory class provides a persistent context for an agent

- Additional capabilities can be added to agents by integrating tools

- ThinkTool enables agents to engage in explicit thinking processes before providing answers

- BeeAI's requirements system provides fine-grained control of agent behavior.

- The ReAct pattern supports reasoning and acting in cycles


### 1. What BeeAI Actually Is
- **BeeAI = a framework for building real-world, production-ready AI agents**
- BeeAI gives you:
    - Structure
    - Control
    - State
    - Tool integration
    - Multi-agent coordination

### 2. Building a Basic AI Conversation
- You create a conversation by:
1. Import modules
2. Initialize model
3. Define messages
4. Run asynchronously
#### 💡 What this really means
- Instead of:
```python
response = model("Hello")
```
- You do:
```python
messages = [
  {"role": "user", "content": "Hello"}
]
await model.run(messages)
```
### 3. Dynamic Prompt Templates
- *“Reusable prompts with variable data”*
#### Why this matters
- Reusable across tasks
- Cleaner architecture
- Easier debugging