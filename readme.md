# AgentHub
## Table Of Contents
- [Concepts](#concepts)
    - [LangChain](#1-langchain)
    - [LangGraph](#2-langgraph)
        - [Key Componnets](#key-components)
        - [Fundamentals](#graph-fundamentals)
        - [How to Build Graph](#graph-construction)
    - [Agentic AI](#3-agentic-ai)
        - [What is Agentic AI](#what-are-ai-agents)
        - [Smart Home Scenario](#one-example-of-an-ai-agent)
## Concepts
### 1. LangChain
#### What is LangChain?
LangChain is an open-source framework that helps developers build applications powered by Large Language Models (LLMs).
Instead of coding every piece manually, LangChain provides ready-made tools, components, and APIs to:

- Connect to language models
- Manage prompts and conversations

- Link external resources (databases, APIs, files, search engines, etc.)

- Build complex workflows by chaining multiple steps together

This makes it easier to move from “just calling an LLM” to building full, production-ready AI apps.

#### ⚙️ Key Capabilities
1. **Tailorable Prompts**:
- You can design prompts dynamically.

- Example: A customer support app could insert the customer’s name, order details, and conversation history into a template before sending it to the LLM.

2. **Chain Link Components**:
- LangChain allows you to connect different steps into a **chain**.
- Example: First step → clean user input → second step → query a database → third step → summarize result.
- This chaining is why it’s called LangChain.

3. **Integration with External Models and Tools**:
- It’s not limited to one LLM (like GPT).

4. **Versatile, Mix-and-Match Components**:
5. **Context Management**:

#### 💡 Why it Matters
- You need **more than just a single LLM call** (e.g., multi-step reasoning, API calls, retrieval from databases).
- You want to integrate AI into real-world applications like chatbots, knowledge bases, assistants, or data pipelines.
- You’re building something **scalable and reusable** instead of quick one-off scripts.
#### 🛠️ Example Applications
- **Chatbots** (customer service, Q&A)
- **Virtual Assistants** (personalized scheduling, task management)
- **Language Tools** (translation, summarization, rewriting)
- Data Apps (RAG systems where the LLM retrieves info from your database or documents)
- Sentiment/Analytics (combining LLM reasoning with ML models and dashboards)

#### Key Components of LangChain
##### components & chains
- **definition**: “components” are small, focused pieces (models, prompts, retrievers, parsers). a chain wires them together so the output of one becomes the input of the next.
- **when to use**: any multi-step flow (clean → prompt → LLM → parse → store).
- **example**:
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant."),
    ("human", "Summarize:\n{text}")
])

chain = prompt | llm  # pipe operator builds a chain

chain.invoke({"text": "LangChain lets you mix & match components to build LLM apps."})

```


##### prompt templates
- **definition**: reusable, parameterized prompts that turn variables into well-structured messages (great for consistency and guardrails).
- **when to use**: you need dynamic content (user name, retrieved docs, tone, format).
- **example**:
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Act as a {role}. Keep answers under {limit} words."),
    ("human", "{question}")
])

msg = prompt.format_messages(role="math tutor", limit="60", question="Explain eigenvalues.")

```

##### vector stores
- **definition**: databases that store **embeddings** (numeric vectors) for semantic search (RAG).
- **when to use**: you want the model to answer using your data (docs, tickets, wikis).
- **example**:
```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

emb = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(collection_name="docs", embedding_function=emb)

# add texts
ids = vectordb.add_texts(["LangChain composes LLM apps.", "RAG adds private knowledge."])
# similarity search
docs = vectordb.similarity_search("How do I add my own data?", k=1)

```
##### example selectors
- **definition**: automatically choose few-shot examples most similar/relevant to the user’s input (boosts accuracy).
- **when to use**:tasks benefit from demonstrations (classification, style transfer, extraction).
- 
##### agents
- **definition**: runtime systems that let LLMs **use tools** (web search, code, DB) and decide which tool to call next (e.g., ReAct-style reasoning). modern LangChain pairs agents with **LangGraph** for reliability, but you can still run simple tool-using chains in pure LangChain.
- **when to use**: multi-step tasks with external actions (search → fetch → compute → answer).
- **example**:
```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You can use tools when needed. Think step-by-step."),
    ("human", "{question}")
])

# Tool-calling chain (the LLM can decide to call `add`)
chain = prompt | llm.bind_tools([add])

result = chain.invoke({"question": "What is 123 + 456?"})
# If the model calls the tool, LangChain will route, execute it, and return a final answer.

```


### 2. Langgraph

#### Key Components
1. **Graph Structure**:
![Graph structure](images/langgraph_graph_structure.png)
Think of your application as a directed graph:
- **Nodes** = **agents** (LLMs, tools, human input nodes, evaluators, etc.).
- **Edges** = **communication channels** (the path data or decisions take from one agent to another).
This setup lets you::
- Chain together specialized agents (e.g., one generates answers, another checks correctness).
- Introduce **cycles** for iterative refinement, such as a loop between a generator and a reviewer until quality criteria are met.
- Visualize workflows in a modular way, like a flowchart for AI reasoning.


2. **State Managemet**:
![Graph State Management](images/langgraph_state_management.png)
- A big challenge in multi-agent systems is keeping track of context across steps. LangGraph solves this with **automatic state management**:
- The **state** contains accumulated messages, facts, or metadata.
- Every time an agent acts, the state is updated — ensuring no history is lost.
- The state can be checkpointed (saved at points in the graph) and even resumed later, which makes long-running or interruptible workflows reliable.

3. **Coordination**:
![Graph Coordination](images/langgraph_coordination.png)
- LangGraph ensures that agents run in the **right order**.
- It handles **conditional routing**, meaning an agent’s output can decide which path to follow (e.g., “If tool needed → Tool Agent, else → Planner”).
- Coordination guarantees that information exchange is seamless (no dropped messages, no out-of-sync agents).

#### Graph Fundamentals
1. **State**:
- **definition**: a shared, evolving memory object that flows through your graph. Every node reads the current state and returns an **updated** state. LangGraph manages persistence/checkpoints so you can resume runs, branch, and inspect history.
- **Typical contents**:
    - `messages`: conversation history (`HumanMessage`, `AIMessage`, etc.)
    - working variables: parsed fields, IDs, partial results
    - control flags: scores, decisions, error info
- **example**:
```python
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: List[BaseMessage]
    facts: dict
    decision: Optional[str]
```

- **TypeDict**:
    - **definition**: TypedDict lets you declare the expected keys and value types of a plain Python dict. It’s a type hint only:
    - **basic syntax**:
    ```python
    from typing import TypedDict, Literal, Optional

    class State(TypedDict):
        name: str
        mood: Literal["happy", "sorrow"]
        note: Optional[str]  # may be None

    ```
    - **usage**:
    ```python
    s: State = {"name": "Alice", "mood": "happy", "note": None}
    ```
    - **example**:
    ```python
    from typing import TypedDict, List
    from langgraph.graph import StateGraph, END

    class LGState(TypedDict):
        messages: List[str]
        decision: str | None

    def node1(s: LGState) -> LGState:
        return {**s, "messages": s["messages"] + ["node1 ran"]}

    def node2(s: LGState) -> LGState:
        # defensive access to avoid KeyError if key might be optional
        msgs = s.get("messages", [])
        return {**s, "messages": msgs + ["node2 ran"], "decision": "done"}

    g = StateGraph(LGState)
    g.add_node("node1", node1)
    g.add_node("node2", node2)
    g.set_entry_point("node1")
    g.add_edge("node1", "node2")
    g.add_edge("node2", END)

    ```
- **Dataclass**
    - **What is a Dataclass in Python?**
        - A **Dataclass** is a decorator (`@dataclass`) provided by Python’s `dataclasses` module.
        - It gives you a simple and concise way to define classes that mainly store data.
        - Instead of writing boilerplate code like `__init__`, `__repr__`, or `__eq__`, Python will auto-generate them for you.
    - **The Example Code Breakdown**
    ```python
    from dataclasses import dataclass

    @dataclass
    class DataclassState:
        name: str
        mood: Literal["happy", "sorrow"]

    ```
    - **The Example Code Breakdown**
    ```python
    from dataclasses import dataclass

    @dataclass
    class DataclassState:
        name: str
        mood: Literal["happy", "sorrow"]

    ```
    - **3. Why Use Dataclass Instead of a Dictionary?**:
    
2. **Edges & Conditional Edges**:
- **Edges** connect nodes and define execution order.
- **Conditional edges** are dynamic branches chosen at runtime from the current state.
- **Cycles** allow iterative refinement (e.g., generate → critique → revise).
- **example**:
```python
from langgraph.graph import StateGraph, END

g = StateGraph(State)
g.add_node("extract", extract_node)
g.add_node("review", review_node)
g.add_node("revise", revise_node)

def route_after_review(state: State) -> str:
    return "revise" if state["decision"] == "needs_changes" else "end"

g.set_entry_point("extract")
g.add_edge("extract", "review")
g.add_conditional_edges("review", route_after_review, {"revise": "revise", "end": END})
g.add_edge("revise", "review")  # cycle until passes review

```
3. **Tool & ToolNode**:
- Tool = any external operation: web/API call, DB query, calculator, custom Python.
- Two common ways to use tools:
    - Call tools inside your own node (most control).
    - Use `ToolNode` to automatically execute tools requested by the LLM.
- example:
```python
# 2) ToolNode pattern (LLM decides which tool to call)
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def fetch_weather(city: str) -> str:
    return f"Sunny today in {city}"

tools = [fetch_weather]
llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def llm_node(state: State) -> State:
    # produce an AIMessage possibly containing tool calls
    ai_msg = llm.invoke(state["messages"])
    state["messages"].append(ai_msg)
    return state

tool_node = ToolNode(tools)  # executes pending tool calls from messages

g = StateGraph(State)
g.add_node("llm", llm_node)
g.add_node("tools", tool_node)
g.set_entry_point("llm")
g.add_edge("llm", "tools")
g.add_edge("tools", "llm")  # loop until no more tool calls

```
4. **Message Types (LangChain / LangGraph)**:
- Messages are structured data units that travel in `state["messages"]`. They capture roles and special behaviors.
    - `HumanMessage` – user input (“What’s the weather in Paris?”).
    - `AIMessage` – LLM output (text and/or tool call instructions).
    - `SystemMessage` – instructions or policy that steer the LLM (“You are a helpful assistant…”).
    - `ToolMessage` – the result of executing a tool, sent back to the LLM to continue reasoning.
    - `RemoveMessage` – instructs the runtime/memory to drop prior messages (useful for context pruning/privacy).
- Typical flow:
    - Add a `HumanMessage`.
    - LLM returns an `AIMessage` (may include a tool call).
    - A tool executes; you add a `ToolMessage` with the result.
    - LLM continues with the new context, producing the final `AIMessage`.
- example:
```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

state["messages"].append(HumanMessage(content="hello world"))
# -> LLM returns AIMessage with tool call
state["messages"].append(ToolMessage(tool_call_id="call-1", content="{'temp': 22}"))

```
5. **react agent**
- Now, we can extend this into a general Agent framework.
- In the Agent router, when we call the model, if it chooses to call a tool, we will return a `ToolMessage` to the user.
- But what happens if we instead pass this `ToolMessage` back into the model?
- **Agent Router**: The Agent Router is responsible for deciding what the model should do with incoming input.
    - If the model needs outside help (like using a calculator, searching the web, or querying a database), it doesn’t just respond directly — it calls a `Tool`.
    - The result of this tool call is wrapped in a `ToolMessage`, which carries the result of that action.

- **What if ToolMessage goes back to the model?** Instead of returning the raw ToolMessage directly to the user, we can feed it back into the LLM.
    - This allows the LLM to **observe the tool’s output** and then **decide the next step**.
    - This enables **multi-step reasoning** (instead of a single question–single answer cycle).
- **The ReAct Paradigm**:
    - ReAct = Reason + Act
    - It’s a general agent framework that combines tool use (actions) with LLM reasoning.
        - **act**: The model issues a command to call a tool (e.g., “search the web”).
        - **observe**: The tool returns results, which the model can read.
        - **reason**: The model thinks about the results and decides what to do next (either call another tool or respond to the user).

6. **Agent and Memory**
- Every step writes a checkpoint:
    - After each advance of the graph, LangGraph saves a **snapshot of state** plus a bit of run metadata.
- Checkpoints live inside a thread
    - A thread is one ongoing session/run (a conversation, a job, etc.). All snapshots for that run are stored on a single timeline.
- You address a run by `thread_id`
    - Pass the same `thread_id` later to resume, continue, or inspect that session.
#### Why Langgraph
1. **Simplified development**
- **What it means**: You describe what should happen (nodes + edges), and LangGraph handles how to run it (order, passing messages, resuming, retries).
- **Why it’s simpler than hand-rolled logic**:

2. **Flexibility**
- **What it means**: You can compose any kind of node and route between them however you like.
- **Ways flexibility shows up**:
    - **Heterogeneous nodes**: LLM agents, tools, evaluators, vector lookups, humans (approval steps), webhooks, your own Python.
3. **Scalability**:
- **What it means**: The runtime is built to handle lots of concurrent, long-running, or complex workflows.
- How you scale in practice:
    - Checkpointing to durable storage: Swap `MemorySaver` for Redis/Postgres-backed checkpointers so any worker can resume a session by thread_id.
    - Horizontal workers: Run multiple stateless worker processes/containers pointing at the same checkpointer. Each takes a turn advancing a graph.
4. **Fault tolerance**:
- **definition**: When something fails, you don’t lose the whole run—and you can intentionally handle failures.
- **Mechanisms & patterns**:
    - **Per-node checkpoints**: On crash/restart, resume from the last finished node with the saved state.

#### Graph Construction
1. **Using the `StateGraph` class**
- LangGraph provides the `StateGraph` class as a base for building graphs.
- We begin by initializing a `StateGraph` using the previously defined State class. This ensures that all nodes
2. **Adding Nodes and Edges**
- Next, we add **nodes** (functional units) and **edges** (connections that determine flow).
- Each node can represent an LLM call, an API call, or a custom function. Edges define how data flows between these nodes.
3. **Special Nodes: START and END**
- **START Node:** A special entry node that represents where the graph begins. User input or initial data is injected into the graph here.
- **END Node:** A special terminal node that indicates where the workflow finishes.
4. **Compiling the Graph**
- Once nodes and edges are added, we **compile** the graph.
- Compiling performs a **basic structure check** to ensure the workflow is valid before execution.

## 3. Agentic AI
### Defining the Spectrum: From AI Agents to Agentic AI
#### What are AI Agents?
#### The Three Capabilities of AI Agents
1. **Autonomy**
- “The ability to function with minimal human intervention after initial deployment…”
- This means once you give the agent a goal, it should:
    - continue working without being manually guided
    - sense new inputs (e.g., new emails)
    - reason about the next step
    - adapt based on context
    - execute actions automatically
2. **Task-Specificity**
- “Each agent is optimized for narrow, well-defined tasks…”
- AI agents are usually **specialists**, not generalists.
- Examples of specialized agents:
    - an email triage agent
    - a database query agent
    - an RAG retrieval agent
    - 

### One example of an AI Agent
![Smart Home Scenario](/images/agent_workflow1.png)
#### 1. **User** → **AI Agent**
- The user asks: **“Latest AI News?”**
- This question goes into the **AI Agent** (the little robot icon on the left).
- Inside the agent, it now has a goal:
    - “Find the latest AI news and tell it to the user.”
- So the agent needs a **plan**:
    1. Search the web for AI news
    2. Read the results
    3. Summarize them
    4. Send a concise answer back
#### 2. **Agent → “AI News” Search Query**
- The grey box labeled “**AI News** 🔍” is the agent deciding what tool call/search query to make.
- The agent converts the user question into a concrete action:
- This is the tool-use step of the agent.
#### 3. “AI News” Query → WWW (the web)
- The **WWW icon** represents the **internet / external environment**.
- The agent (through a tool) sends the search query to the web:
    - Google/Bing/News API, etc.
- The web responds with:
    - news articles
    - titles
    - snippets
    - descriptions
- This is the **perception** step: the agent is gathering raw information from the environment.
#### 4. WWW → Raw News Content
- The curved arrow from the WWW back down to the grey text box (e.g., “Tech company unveils…”)
- represents:
    - The web returns **news articles**
    - 

### Agentic AI
#### What is Agentic AI
#### 1. Task Decomposition
- **Meaning**:
    - Agentic AI automatically breaks a large goal into smaller, solvable subtasks.
- **Example**:
    - User says:
        - “Plan my vacation.”
    - A single-task agent might fail.
    - But an agentic system breaks it into:
        1. Book flights 
        2. Find hotels
        3. Check visa requirements
        4. Compare prices
        5. Generate itinerary
        
#### 2. Inter-Agent Communication
- **Meaning**:
    - Multiple agents communicate with each other—like digital coworkers in Slack.
- **Example**
    - Flight agent → sends booked flight details to itinerary agent
    - Hotel agent → sends hotel details to pricing agent
    - Visa agent → sends visa info to scheduling agent
- **Why this matters**:
    - Allows collaborative problem solving
    - Enables division of labor
    - Allows agents to adapt based on each other's output
    - Makes the system dynamically intelligent
- **How it relates to your Reflect Agent:**
    - You currently have 1 agent (planner/reflecting agent).
    - In a multi-agent setup:
        - Task Planner → decomposes
        - Search Agent → fetches info
        - Summarizer Agent → writes summary
        - Reflection Agent → checks mistakes
    - You can easily expand into this pattern.
#### 3. Memory and Reflection

- Agentic AI can:
- **Remember past steps**:
    - (What it already tried, what worked, what failed)
- **Reflect on its output**:
    - (“Is this correct?”, “Should I fix this?”, “Should I retry?”)
- **Learn from mistakes**