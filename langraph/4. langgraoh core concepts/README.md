# LangGraph Core Concepts — Revision Notes
*(Video 4 of Agentic AI playlist — "LangGraph Core Concepts")*

> Teacher's structure: (1) LLM workflows & 5 common patterns, (2) Graphs/Nodes/Edges,
> (3) State, (4) Reducers, (5) Execution model (Pregel/superstep). This file
> keeps that structure and adds concrete syntax + missing context.

---

## 1. Quick Recap: What is LangGraph?

> An orchestration framework for building intelligent, stateful, multi-step
> LLM workflows.

Core capabilities recap: represents a workflow as a **graph** (nodes = tasks,
edges = execution order), supports **parallel execution, loops, branching,
memory, and resumability**. Ideal for agentic + production-grade AI apps.

---

## 2. LLM Workflows

**Workflow** = a series of tasks executed in order to achieve a goal.
**LLM workflow** = a workflow where one or more steps depend on an LLM
(prompting, reasoning, tool calling, memory access, decision making).

Workflows can be **linear, parallel, branched, or looped**.

### The 5 Common LLM Workflow Patterns

> 🔧 Gap-fill: these five patterns are not the teacher's own invention — they
> are, near-verbatim, the patterns from **Anthropic's "Building Effective
> Agents"** blog post (the same one referenced in Video 3 for the
> workflow-vs-agent distinction). Worth reading directly since the video
> presents them without that framing this time. Link: anthropic.com/research/building-effective-agents

| # | Pattern | What it does | Example |
|---|---|---|---|
| 1 | **Prompt Chaining** | Break a complex task into a fixed sequence of LLM calls, each feeding the next. Can insert a "gate" check between steps. | Topic → outline (LLM 1) → detailed report (LLM 2), with a word-count check in between |
| 2 | **Routing** | An LLM classifies the input and directs it to one of several specialized downstream LLMs/paths. | Customer support query → routed to refund-bot / tech-bot / sales-bot |
| 3 | **Parallelization** | Split a task into independent subtasks, run them **simultaneously**, then aggregate results. Subtask *types* are fixed in advance. | Content moderation: check community-guidelines, misinformation, and sexual-content in parallel, then aggregate |
| 4 | **Orchestrator–Worker** | Like parallelization, but the subtasks are **decided dynamically** at runtime by an orchestrator LLM — not fixed beforehand. | Research assistant: orchestrator decides whether to search Google Scholar vs Google News depending on the query, and assigns work to worker LLMs accordingly |
| 5 | **Evaluator–Optimizer** | A generator LLM produces a solution; an evaluator LLM checks it against criteria and either accepts or rejects with feedback; loops until accepted. | Draft an email/blog → evaluate → regenerate with feedback → repeat until good |

> 🔧 Gap-fill — Parallelization vs Orchestrator-Worker is the pair students
> usually confuse. The one-line test: **if you can name the subtasks before
> running the workflow, it's Parallelization; if the subtasks depend on the
> input and are decided at runtime, it's Orchestrator-Worker.**

---

## 3. Graphs, Nodes & Edges

> "The most important core concept in LangGraph" — per the teacher.

**Worked example:** a UPSC essay-practice website.
Flow: generate topic → user writes essay → evaluate on 3 criteria (clarity,
depth, language) → score out of 15 → if ≥ threshold, congratulate; else give
feedback and loop back to essay-writing if the user wants to retry.

- **Node** = a single task in the workflow. Behind the scenes, **every node
  is just a plain Python function.**
- **Edge** = tells LangGraph *when* to execute the next node (control flow).
  Edges can be:
  - **Sequential** — one after another
  - **Parallel** — multiple nodes fire at once
  - **Conditional** — branches based on a condition
  - **Looping** — cycles back to an earlier node

> Nodes = **what** to do. Edges = **when/what next** to do.

> 🔧 Gap-fill — the actual code for the UPSC example (video stays conceptual,
> never shows this):
> ```python
> from langgraph.graph import StateGraph, END
>
> graph = StateGraph(EssayState)
> graph.add_node("generate_topic", generate_topic_fn)
> graph.add_node("collect_essay", collect_essay_fn)
> graph.add_node("evaluate", evaluate_fn)        # scores clarity/depth/language
>
> graph.set_entry_point("generate_topic")
> graph.add_edge("generate_topic", "collect_essay")
> graph.add_edge("collect_essay", "evaluate")
>
> def route_after_eval(state) -> str:
>     if state["total_score"] >= 10:
>         return "pass"
>     return "retry" if state["wants_retry"] else "fail"
>
> graph.add_conditional_edges(
>     "evaluate",
>     route_after_eval,
>     {"pass": END, "fail": END, "retry": "collect_essay"}
> )
>
> app = graph.compile()
> ```

---

## 4. State

> "In LangGraph, state is a shared memory that flows through your workflow.
> It holds all the data being passed between nodes as your graph runs."

- State = the pieces of data a workflow **needs** to execute, which **evolve
  over time** as execution proceeds (e.g. essay text, clarity score, depth
  score, language score, overall score).
- State is:
  1. **Shared** — every node has access to the full state.
  2. **Mutable** — any node can change it.
- Built as a **`TypedDict`** (a Python class) — or a Pydantic model.

> 🔧 Gap-fill — actual syntax (video only says "it's a typed dictionary,
> nothing special" without showing it):
> ```python
> from typing import TypedDict, Optional
>
> class EssayState(TypedDict):
>     topic: str
>     essay_text: str
>     clarity_score: float
>     depth_score: float
>     language_score: float
>     total_score: float
>     wants_retry: bool
> ```
> Each node receives the full state as its argument and returns a dict of
> the fields it wants to update — LangGraph merges that into the shared state
> before passing it to the next node.

---

## 5. Reducers

> Closely tied to state's "mutable" property.

**The problem:** by default, when a node returns a value for a state key,
LangGraph **overwrites/replaces** the old value with the new one. This is
fine for something like a running `result` in a calculator workflow, but
**breaks** in scenarios like a chatbot:

- Human: "Hi, my name is Nitish" → state: `messages = "Hi, my name is Nitish"`
- LLM replies → state: `messages = "Hi, how can I help you?"` (overwrote it!)
- Human: "What's my name?" → LLM has **no memory** of the name — it was erased.

**The fix — Reducers:** a reducer defines **how** an update is applied to a
state key: **replace, append/add, or merge**. Each key in the state can have
its own reducer.

> For the chatbot case, instead of replacing, you want to **append** every
> new message to a running list — that's the correct policy for a
> `messages` field. Same idea for the UPSC essay example if the student wants
> to review **all** of their past essay attempts, not just the latest one —
> you'd append instead of overwrite there too.

> 🔧 Gap-fill — actual syntax (video describes reducers conceptually but
> never shows the `Annotated` + `operator.add` pattern, which is the
> standard way to declare one):
> ```python
> from typing import TypedDict, Annotated
> from operator import add
> # For chat-style message lists, LangGraph ships a purpose-built reducer:
> from langgraph.graph.message import add_messages
>
> class ChatState(TypedDict):
>     messages: Annotated[list, add_messages]   # appends instead of overwriting
>
> class EssayHistoryState(TypedDict):
>     essay_attempts: Annotated[list, add]      # generic append reducer
>     total_score: float                        # no annotation = default overwrite behavior
> ```
> `add_messages` (LangGraph's built-in reducer for chat) is smarter than a
> plain list-append — it also handles de-duplication and updates to existing
> messages by ID. This is the reducer you'll actually use for any chatbot
> node — the video never names it even though it's the standard tool for the
> exact chatbot problem it describes.

**Rule of thumb:** reducers matter most in **parallel workflows**, where
multiple nodes may update the same state key at the same time and you need
to control how those updates combine (overwrite vs. append vs. merge).

---

## 6. Execution Model (Pregel-style / Superstep)

> 🔧 Gap-fill — big context the video only mentions in passing: LangGraph's
> execution model is explicitly inspired by **Google Pregel**, a
> large-scale distributed graph-processing system used internally at Google
> (the same "bulk synchronous parallel" model behind systems like Apache
> Giraph). Knowing this name helps if you ever read LangGraph's internals or
> academic references to it.

### Two Phases

**Phase 1 — Graph Definition & Compilation**
1. **Define** — declare nodes, edges, and the state schema (TypedDict).
2. **Compile** — `graph.compile()` validates the graph's structure: e.g.
   checks there's no **orphan node** (a node with no incoming/outgoing edge
   connecting it to the rest of the graph).

**Phase 2 — Execution**
1. **Invocation** — the initial state is passed to the first node via
   `app.invoke(initial_state)`. This **activates** the node — its underlying
   Python function is called.
2. The node runs, then applies a **partial update** to the shared state.
3. That updated state travels along an edge to the next node(s) — this
   hand-off is called **message passing**.
4. Each round of node-activation is called a **superstep** — not "step,"
   because a single round can contain **more than one node executing in
   parallel** (e.g. 3 nodes firing at once after a fan-out edge). A superstep
   can be 1 step or many steps run concurrently.
5. When results converge from parallel nodes, **reducers** decide how those
   concurrent updates merge into the shared state.
6. **Termination:** execution stops when there is no active node left and no
   edge is currently passing a message.

> 🔧 Gap-fill — you never manually call node 1, then node 2, etc. You call
> `app.invoke()` once on the compiled graph; internal message-passing +
> superstep scheduling handles the rest.
> ```python
> app = graph.compile()
> result = app.invoke({"topic": "", "essay_text": "", ...})  # runs to completion
> print(result)   # final state after all supersteps finish
> ```

### Terminology Summary

| Term | Meaning |
|---|---|
| Graph definition | Declaring nodes, edges, and state schema |
| Compile | Validates graph structure (e.g. catches orphan nodes) before it can run |
| Invocation | Passing the initial state into the first node to start execution |
| Message passing | Sending the (partially updated) state along an edge to the next node(s) |
| Superstep | One round of execution — may contain 1 or several parallel node activations |
| Termination | No active nodes + no in-flight messages on edges |

---

## 7. Self-Test (revision)

1. What's the one-line test to distinguish Parallelization from Orchestrator-Worker? *(subtasks known in advance vs. decided dynamically at runtime)*
2. What are the two defining properties of LangGraph's state object? *(shared across all nodes, mutable)*
3. What data structure typically backs a LangGraph state? *(TypedDict, or Pydantic model)*
4. Why does a default (overwrite) update break a chatbot workflow? *(each new message replaces the last instead of accumulating chat history)*
5. What's the built-in reducer for chat messages in LangGraph? *(`add_messages`)*
6. Why is a "step" called a "superstep" in LangGraph's execution model? *(a single round can include multiple nodes executing in parallel, not just one)*
7. What does `compile()` check for before execution? *(structural validity of the graph — e.g. no orphan nodes)*
8. What system inspired LangGraph's execution model? *(Google Pregel)*