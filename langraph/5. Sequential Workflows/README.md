# Sequential Workflows in LangGraph — Revision Notes
*(Video 5 of Agentic AI playlist — first hands-on coding video)*

> Teacher's structure: (1) setup/installation, (2) non-LLM sequential
> workflow (BMI calculator), (3) extended BMI workflow (add a 2nd node),
> (4) simplest LLM workflow (Q&A), (5) prompt chaining workflow (blog
> generator). This file keeps that structure, fills in the exact code, and
> adds what the teacher skipped or rushed past.

---

## 0. Recap — Where We Are

Videos 1–4 were theory: Generative vs Agentic AI → what Agentic AI is →
why LangGraph exists (vs LangChain) → LangGraph's core concepts (workflows,
graphs/nodes/edges, state, reducers, execution model). **This is the first
practical/coding video.**

---

## 1. Project Setup

```bash
mkdir langgraph-tutorials
cd langgraph-tutorials
# open in VS Code

python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # macOS/Linux (video is on Windows, only shows this one)

pip install langgraph
pip install langchain
pip install langchain-openai
pip install python-dotenv
```

> 🔧 Gap-fill — why each package:
> - `langgraph` → the orchestration framework itself
> - `langchain` → still needed for LLM-related components (chat models,
>   prompt templates, document loaders, text splitters) — LangGraph doesn't
>   reimplement these, it just orchestrates them (per Video 3)
> - `langchain-openai` → OpenAI-specific chat model wrapper (`ChatOpenAI`)
> - `python-dotenv` → reads environment variables (API keys) from a `.env` file

**`.env` file** (create in project root, never commit this):
```
OPENAI_API_KEY=sk-...your-key-here...
```

**Why Jupyter Notebook (`.ipynb`) and not plain `.py` files:**
LangGraph's graph-visualization code (see §3 below) only renders inside a
notebook — it needs a rich display environment (renders as an image via
IPython's display system). Plain `.py` scripts can't show it. For actual
projects later in the playlist, the teacher will switch to `.py` files.

**Test the install** — new notebook `0_test_installation.ipynb`:
```python
from langgraph.graph import StateGraph
```
If this imports without error (and your editor shows autocomplete
suggestions for `langgraph`/`langchain`), the environment is set up correctly.

---

## 2. The 4-Step Recipe for Any LangGraph Workflow

Every workflow in LangGraph follows the same four steps:

1. **Define State** — a `TypedDict` describing the data the workflow needs
2. **Define Graph** — create a `StateGraph` object, add nodes, add edges
3. **Compile** — `graph.compile()` — validates the graph structure
4. **Execute** — `workflow.invoke(initial_state)` — runs it end-to-end

Keep this checklist in mind — every example below follows it exactly.

---

## 3. Example 1 — BMI Calculator (non-LLM, single-node workflow)

**Why start with a non-LLM example:** keeps focus purely on LangGraph syntax,
without LLM response variability/latency getting in the way.

**Workflow:** input `weight` + `height` → one node computes `bmi` → output.

**State:**
```python
from typing import TypedDict

class BMIState(TypedDict):
    weight: float   # kg
    height: float   # meters
    bmi: float
```

**Node function** — receives state, returns (partial) state:
```python
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]
    bmi = weight / (height ** 2)
    state["bmi"] = round(bmi, 2)
    return state
```

**Graph — build, wire, compile:**
```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(BMIState)
graph.add_node("calculate_bmi", calculate_bmi)

graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", END)

workflow = graph.compile()
```

> 🔧 Gap-fill — `START` and `END` are **not** real nodes you write functions
> for; they're sentinel/placeholder markers imported straight from
> `langgraph.graph` that tell LangGraph where the graph begins and
> terminates.

**Visualize the graph** (Jupyter-only — uses IPython display):
```python
from IPython.display import Image, display
display(Image(workflow.get_graph().draw_mermaid_png()))
```
> 🔧 Gap-fill — the video pastes this from LangGraph's docs without
> explaining it: `.get_graph()` returns the graph's structure, and
> `.draw_mermaid_png()` renders it as a Mermaid-diagram PNG. This is exactly
> why the whole video is done in Jupyter — this call only renders inline in
> a notebook.

**Run it:**
```python
initial_state = {"weight": 80, "height": 1.73}
final_state = workflow.invoke(initial_state)
print(final_state)
# {'weight': 80, 'height': 1.73, 'bmi': 26.74}
```

> **Key rule (repeated from earlier videos, now seen in code):** a compiled
> graph always takes a **state dict in**, and always returns a **state dict
> out** — never just a bare value.

---

## 4. Example 1b — Extending BMI Calculator (2-node sequential workflow)

Add a second node that labels the BMI category (underweight / normal /
overweight / obese).

**Changes needed (checklist form — the video does these ad hoc):**

1. Add a new field to state:
```python
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str          # NEW
```

2. Add a new node function:
```python
def label_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif bmi < 25:
        state["category"] = "Normal"
    elif bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"
    return state
```

3. Register the node and rewire the edges:
```python
graph.add_node("label_bmi", label_bmi)

graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")   # was: calculate_bmi -> END
graph.add_edge("label_bmi", END)                # new final edge
```

> 🔧 Gap-fill — note you don't need to remove the old
> `add_edge("calculate_bmi", END)` call manually if you're rebuilding the
> graph object from scratch, but if you're editing in a notebook cell that
> already ran, you must physically delete/replace that old edge line —
> otherwise `calculate_bmi` would point to **both** `label_bmi` and `END`,
> which is invalid for a simple `add_edge` (that ambiguity is what
> conditional edges exist to handle — see Video 3/4 notes).

Result: `START → calculate_bmi → label_bmi → END`.

---

## 5. Example 2 — Simplest LLM Workflow (Q&A)

**Workflow:** `question` in → one node calls the LLM → `answer` out.

**Setup:**
```python
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI()
```

**State:**
```python
class LLMState(TypedDict):
    question: str
    answer: str
```

**Node:**
```python
def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]
    prompt = f"Answer the following question: {question}"
    answer = model.invoke(prompt).content
    state["answer"] = answer
    return state
```

**Graph + run:**
```python
graph = StateGraph(LLMState)
graph.add_node("llm_qa", llm_qa)
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

workflow = graph.compile()

initial_state = {"question": "How far is the moon from the earth?"}
final_state = workflow.invoke(initial_state)
print(final_state["answer"])
```

> ⚠️ **Teacher's own caveat, worth remembering:** this exact result could be
> gotten with a single line — `model.invoke(prompt).content` — no LangGraph
> needed. Using LangGraph here is deliberately overkill ("**ghumaa ke kaan
> pakadna**" — "reaching around to grab your ear the long way"), purely to
> teach syntax. The real payoff shows up once workflows get non-linear
> (branches, loops, multi-agent) — which starts in upcoming videos.

> 🔧 Gap-fill — `.content` is needed because `model.invoke(...)` returns a
> full **`AIMessage`** object (with metadata like token usage, response
> `id`, etc.), not a plain string — `.content` extracts just the text.

---

## 6. Example 3 — Prompt Chaining Workflow (Blog Generator)

Recall from Video 4: **Prompt Chaining** = a fixed sequence of LLM calls,
each one's output feeding the next.

**Workflow:** `title` → Node 1 generates an `outline` → Node 2 generates
`content` (the full blog) using both title and outline.

**State:**
```python
class BlogState(TypedDict):
    title: str
    outline: str
    content: str
```

**Node 1 — create outline:**
```python
def create_outline(state: BlogState) -> BlogState:
    title = state["title"]
    prompt = f"Generate a detailed outline for a blog on the topic: {title}"
    outline = model.invoke(prompt).content
    state["outline"] = outline
    return state
```

**Node 2 — create blog from title + outline:**
```python
def create_blog(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]
    prompt = f"Write a detailed blog on the title '{title}' using the following outline:\n{outline}"
    content = model.invoke(prompt).content
    state["content"] = content
    return state
```

**Graph + run:**
```python
graph = StateGraph(BlogState)
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)

workflow = graph.compile()

initial_state = {"title": "Rise of AI in India"}
final_state = workflow.invoke(initial_state)

print(final_state["title"])
print(final_state["outline"])
print(final_state["content"])
```

### Why this is better than a plain LangChain chain (the key takeaway)

With a normal LangChain chain, only the **final** output survives — you'd
get the finished blog but **lose access to the intermediate outline**
(mirrors the exact "chains are stateless" problem from Video 3). With
LangGraph, because everything lives in **shared, evolving state**, the
*final* state still contains `title`, `outline`, **and** `content` — every
intermediate artifact stays accessible after the run finishes.

---

## 7. Homework (from the video) — with a worked solution

**Task:** add a third node, `evaluate_blog`, that scores the blog against
its outline (an integer score), after `create_blog`.

> 🔧 Gap-fill — the video states the task but doesn't solve it. Here's a
> complete implementation matching the same patterns used above:

```python
class BlogState(TypedDict):
    title: str
    outline: str
    content: str
    score: int          # NEW

def evaluate_blog(state: BlogState) -> BlogState:
    outline = state["outline"]
    content = state["content"]
    prompt = (
        f"Based on this outline:\n{outline}\n\n"
        f"Rate the following blog on a scale of 1 to 10 for how well it "
        f"follows the outline. Respond with ONLY an integer.\n\nBlog:\n{content}"
    )
    raw_score = model.invoke(prompt).content
    state["score"] = int(raw_score.strip())
    return state

graph.add_node("evaluate_blog", evaluate_blog)
graph.add_edge("create_blog", "evaluate_blog")   # replaces create_blog -> END
graph.add_edge("evaluate_blog", END)
```

> ⚠️ Practical note not covered in the video: LLMs don't always return a
> clean integer (they might say "Score: 8" or "8/10"). In production you'd
> want to either strengthen the prompt (e.g. "respond with ONLY the digit,
> nothing else") or parse the response more defensively (e.g. regex-extract
> the first number) before `int(...)`, to avoid a `ValueError` crash.

---

## 8. Self-Test (revision)

1. What are the 4 steps every LangGraph workflow follows? *(define state → define graph/nodes/edges → compile → invoke)*
2. Why does the graph visualization code only work in Jupyter, not `.py` files? *(it uses IPython's `display()`/image rendering, which needs a notebook's rich display environment)*
3. What do `START` and `END` represent? *(sentinel markers imported from `langgraph.graph`, not real nodes with functions)*
4. What does a compiled graph always take as input and return as output? *(a state dict, both ways — never a bare value)*
5. Why use `.content` after `model.invoke(prompt)`? *(invoke returns a full AIMessage object; `.content` extracts just the text)*
6. In the blog example, what's the concrete advantage of LangGraph's state over a LangChain chain's output? *(the outline stays accessible in the final state, not just the finished blog)*
7. What's the LangGraph term for the workflow pattern used in the blog generator? *(Prompt Chaining — from Video 4's five common patterns)*