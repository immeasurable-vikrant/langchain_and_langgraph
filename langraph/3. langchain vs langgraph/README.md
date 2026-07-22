# LangGraph Intro — Revision Notes
*(Video 3 of Agentic AI playlist — "Why LangGraph exists")*

> Teacher's structure: 8 challenges of building a complex non-linear workflow
> (Automated Hiring example) in **LangChain**, and how **LangGraph** solves each.
> This file keeps that structure but adds the concrete syntax, terms, and
> details the video only gestured at.

---

## 1. Quick Recap: What is LangChain?

> Open-source library to simplify building LLM-based applications.

**Building blocks (modular components):**

| Component | Purpose |
|---|---|
| **Model** | Unified interface to talk to any LLM provider (OpenAI, Anthropic, HuggingFace, Ollama) |
| **Prompts** | Prompt engineering / templating |
| **Retrievers** | Fetch relevant docs from a vector store / knowledge base (RAG) |
| **Chains** | Link components so output of one = input of next |
| **Tools** | Let an LLM call external functions/APIs (basic agent behavior) |

**What you can build with just LangChain:** chatbots, summarizers, multi-step
linear pipelines, RAG apps, and *basic* tool-calling agents.

**Its core limitation:** Chains are fundamentally **linear** (left → right, no
stopping). That's the seed of every problem discussed below.

---

## 2. Workflow vs. Agent — the distinction the video leaned on

Sourced from Anthropic's blog *"Building Effective Agents"* (worth reading directly):

- **Workflow** = LLMs and tools orchestrated through **predefined code paths**.
  The developer draws the flowchart in advance. Same input → same path, every time.
- **Agent** = the LLM **dynamically decides** its own steps and tool usage,
  maintaining control over how it accomplishes a task. Path is not fixed in advance.

> ⚠️ Gap-fill: the video's "Automated Hiring" flowchart is a **workflow**, not
> an agent — even though LLMs are used inside it. Don't confuse "uses an LLM"
> with "is agentic." A true agentic version of hiring would let the LLM itself
> decide *whether* to post on LinkedIn vs Naukri, *whether* to wait 7 days or
> 3, etc. — the human/developer isn't pre-scripting the graph.

---

## 3. The 8 Challenges — LangChain vs LangGraph

### Challenge 1 — Control Flow Complexity (branches, loops, jumps)

- LangChain chains are linear; no native `if/else`, `while`, or `goto` construct.
- Anything non-linear = you write raw Python around the chain ("**glue code**").
- More glue code → harder to maintain, debug, and collaborate on.

**LangGraph's fix:** Represent the whole workflow as a **graph** — each task is
a **node** (a plain Python function), and **edges** define control flow,
including **conditional edges** (branches) and **cycles** (loops) — natively,
with zero glue code.

> 🔧 Gap-fill — the actual syntax skipped in the video:
> ```python
> from langgraph.graph import StateGraph, END
>
> graph = StateGraph(HiringState)          # state schema, see Challenge 2
> graph.add_node("hiring_request", hiring_request_fn)
> graph.add_node("create_jd", create_jd_fn)
> graph.add_node("check_approval", check_approval_fn)   # not a real node itself;
> graph.add_node("post_jd", post_jd_fn)                 # approval is a *routing function*
>
> graph.set_entry_point("hiring_request")
> graph.add_edge("hiring_request", "create_jd")
>
> # Conditional edge: routing function returns a STRING key,
> # which is mapped to the next node.
> def route_after_jd(state) -> str:
>     return "approved" if state["jd_approved"] else "rejected"
>
> graph.add_conditional_edges(
>     "create_jd",
>     route_after_jd,
>     {"approved": "post_jd", "rejected": "create_jd"}   # "rejected" -> loop back
> )
> graph.add_edge("post_jd", END)
>
> app = graph.compile()
> app.invoke({"hiring_prompt": "..."})
> ```
> Note: `check_approval` in the video is really a **routing/conditional function**,
> not a separate node — a small but important distinction the transcript blurs.

---

### Challenge 2 — State Handling

- **State** = the set of key-value data points that evolve as the workflow
  progresses (jd, jd_approved, jd_posted, num_applications, shortlist, offer_status…).
- LangChain has no built-in mechanism to track structured state like this.
  It only has **conversational memory** (chat history), which is not the same thing.
- Workaround = a manual global dict passed and mutated by hand through the
  chain → fragile, error-prone.

**LangGraph's fix:** Execution is **stateful**. You define a schema for the
state (TypedDict or Pydantic model), and **every node** receives the current
state as input and returns updates to it.

> 🔧 Gap-fill — the schema syntax:
> ```python
> from typing import TypedDict, Optional
>
> class HiringState(TypedDict):
>     jd: Optional[str]
>     jd_approved: bool
>     jd_posted: bool
>     num_applications: int
>     min_applications: int
>     shortlisted: list
>     offer_status: Optional[str]
>
> def create_jd_fn(state: HiringState) -> HiringState:
>     jd_text = llm.invoke(...)
>     return {"jd": jd_text}   # LangGraph MERGES this into the existing state
> ```
> By default LangGraph **overwrites** a key with whatever the node returns for
> that key (unless you define a custom "reducer" via `Annotated[...]`, e.g. to
> append to a list instead of replacing it — the video never mentions
> reducers, but they matter a lot once you have list/counter fields like
> `num_applications` or `shortlisted`).

---

### Challenge 3 — Event-Driven Execution (pause & resume)

- Sequential execution = runs start-to-finish without stopping.
- Event-driven execution = the workflow **pauses**, waits for an external
  trigger (7 days passing, a candidate's email reply), then **resumes**.
- LangChain was designed for short-lived, sequential chains — no native
  pause/resume. Workaround = split into multiple chains + external Python
  code to track time/state transfer → more glue code.

**LangGraph's fix:** Because execution is stateful, you can **checkpoint**
(save) state at any node and resume later from an external trigger.

> 🔧 Gap-fill — what a "checkpointer" actually is (the video keeps referencing
> it without defining it clearly):
> A checkpointer is a **persistence backend** LangGraph writes the state to
> after every node executes. Options range from `MemorySaver` (in-memory, lost
> on restart — fine for dev) to `SqliteSaver` / `PostgresSaver` (durable, for
> production). It's tied to a **`thread_id`** — a unique ID per workflow run
> — so you can have many hiring processes running independently and resume
> the correct one.
> ```python
> from langgraph.checkpoint.memory import MemorySaver
>
> checkpointer = MemorySaver()
> app = graph.compile(checkpointer=checkpointer)
>
> config = {"configurable": {"thread_id": "hiring-run-42"}}
> app.invoke({"hiring_prompt": "..."}, config=config)
>
> # ... 7 days later, external trigger fires ...
> app.invoke(None, config=config)   # resumes from last checkpoint of thread 42
> ```

---

### Challenge 4 — Fault Tolerance

- Two fault types: **small** (node-level, e.g. a LinkedIn API call fails) and
  **big** (system-level, e.g. the server hosting the workflow crashes).
- LangChain has no fault tolerance — a crash mid-chain means starting over
  from step 1 (chains are assumed to be short-lived).

**LangGraph's fix:**
- Small faults → **retry logic** (retry a node call after a delay/backoff).
- Big faults → **recovery via checkpoints**: since state is saved after every
  node, you can resume exactly where the crash happened instead of restarting.

> 🔧 Gap-fill — retry isn't automatic by default; you configure it per node:
> ```python
> from langgraph.types import RetryPolicy
>
> graph.add_node("post_jd", post_jd_fn, retry=RetryPolicy(max_attempts=3))
> ```

---

### Challenge 5 — Human-in-the-Loop (HITL)

- Some steps need a human decision (approve JD, approve posting) before
  continuing — sometimes with a delay of hours/days.
- LangChain can ask for input mid-chain, but only for **short waits** —
  a chain sitting idle for 24 hours burns compute and risks crashing.
  Workaround = manually split the chain in two + manually re-pass state.

**LangGraph's fix:** HITL is a **first-class feature**, built on the same
checkpoint mechanism — the graph can pause **indefinitely** (minutes to days)
waiting for human input, then resume from that exact point.

> 🔧 Gap-fill — the actual mechanism is the `interrupt()` function (not shown
> in the video at all, just implied conceptually):
> ```python
> from langgraph.types import interrupt, Command
>
> def check_approval_node(state: HiringState) -> HiringState:
>     decision = interrupt({"jd_for_review": state["jd"]})  # pauses graph here
>     return {"jd_approved": decision == "approve"}
>
> # Resuming after a human responds, days later:
> app.invoke(Command(resume="approve"), config=config)
> ```
> This is the piece that actually makes the "pause indefinitely, resume later"
> claim concrete — the video only quotes the docs' definition without showing
> how it's coded.

---

### Challenge 6 — Nested Workflows (Subgraphs) *(a feature, not a "challenge")*

- A **subgraph** = a graph used as a single node inside another graph
  (encapsulation applied to LangGraph).
- Two big use cases:
  1. **Multi-agent systems** — e.g. self-driving car example: sensor-fusion
     agent, driving agent, entertainment agent, orchestrator agent.
  2. **Reusability** — build one small graph (e.g. "get approval") once, reuse
     it at multiple points in a larger graph, like a reusable function.
- LangChain has no equivalent concept.

> 🔧 Gap-fill — minimal subgraph pattern:
> ```python
> approval_subgraph = StateGraph(ApprovalState)
> # ... add nodes/edges to approval_subgraph ...
> approval_app = approval_subgraph.compile()
>
> main_graph.add_node("get_jd_approval", approval_app)  # a compiled graph IS a valid node
> ```
> When state schemas differ between parent and subgraph, you typically need a
> small adapter function to map parent state → subgraph state and back — the
> video flags this as "something to learn later" but it's worth knowing the
> keyword now: **state key mapping / shared state keys**.

---

### Challenge 7 — Observability

- Observability = how easily you can monitor, debug, and understand what a
  workflow did at runtime — critical in production (auditing costs, weird
  agent decisions, debugging).
- **LangSmith** integrates with LangChain, but it can only see LangChain's own
  calls (e.g. an LLM call) — it's blind to your custom glue code (your `while`
  loops, manual dict handling). Result: **partial observability**.

**LangGraph's fix:** Since there's no glue code — everything (node transitions,
state diffs before/after each node, human-in-the-loop decisions, message
exchanges) happens *inside* the graph — LangSmith gets **full observability**
when paired with LangGraph.

> 🔧 Gap-fill — enabling this is just environment variables, not extra code:
> ```bash
> export LANGCHAIN_TRACING_V2=true
> export LANGCHAIN_API_KEY=your_key
> export LANGCHAIN_PROJECT=hiring-workflow
> ```
> Once set, every `app.invoke()` call is automatically traced to LangSmith —
> the video implies "tight integration" but doesn't mention this is basically
> zero-code setup.

---

## 4. Summary Definitions (verbatim-equivalent, for recall)

> **LangGraph** is an orchestration framework that enables you to build
> stateful, multi-step, and event-driven workflows using LLMs. It's ideal for
> designing both single-agent and multi-agent agentic AI applications.

**Mental model:** LangGraph = a flowchart engine for LLMs.
- Steps → **nodes**
- Connections → **edges**
- Decision logic → **conditional edges**
- LangGraph handles: state management, branching, looping, pausing/resuming,
  fault recovery.

---

## 5. When to Use What

| Use **LangChain** when… | Use **LangGraph** when… |
|---|---|
| Simple linear workflow | Complex non-linear workflow |
| Prompt chain, summarizer, basic RAG | Needs conditionals / loops |
| Short-lived execution | Needs human-in-the-loop over long delays |
| | Needs multi-agent coordination |
| | Needs async / event-driven execution |

**Important:** LangGraph is **built on top of** LangChain, not a replacement.
You still use LangChain's components (chat models, prompt templates,
retrievers, document loaders, text splitters, tools) *inside* LangGraph nodes.
LangGraph only replaces LangChain's **chain** mechanism for orchestration —
not its component library.

---

## 6. Terms Introduced (glossary)

| Term | Meaning |
|---|---|
| Glue code | Custom Python written outside a framework's constructs to stitch logic together — a smell that the framework doesn't fit the use case |
| State | The evolving set of key-value data a workflow depends on |
| Checkpointer | Persistence layer that saves graph state after each node, keyed by `thread_id` |
| `interrupt()` | LangGraph primitive that pauses a graph indefinitely for human input |
| Subgraph | A compiled graph used as a node inside a larger graph |
| Reducer | Custom merge logic for a state field (e.g. append vs overwrite) — not covered in video, worth knowing |
| Routing / conditional function | A function attached via `add_conditional_edges` that returns a string key deciding the next node |

---

## 7. Self-Test (revision)

1. What's the core structural difference between a workflow and an agent? *(predefined code paths vs. LLM-directed dynamic paths)*
2. Why can't LangChain natively express loops/branches? *(chains are linear by design)*
3. What two things does a checkpointer let you do? *(fault recovery + long-pause human-in-the-loop)*
4. What LangGraph primitive implements human-in-the-loop pauses? *(`interrupt()` + `Command(resume=...)`)*
5. Name the two use cases for subgraphs. *(multi-agent systems, reusability)*
6. Why is LangSmith's observability "partial" with LangChain but "full" with LangGraph? *(glue code is invisible to LangSmith; LangGraph has no glue code)*
7. True/False: LangGraph replaces LangChain. *(False — it replaces chains for orchestration; components are still LangChain's)*