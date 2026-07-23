# Parallel Workflows in LangGraph — Revision Notes
*(Video 6 of Agentic AI playlist — parallel workflows, structured output, reducers)*

> Teacher's structure: (1) simple non-LLM parallel workflow (Cricket
> batsman stats) → hits the "partial update" problem live → (2) a harder
> LLM-based parallel workflow (UPSC essay evaluator) combining parallel
> nodes + structured output + reducers. This file keeps that structure and
> fills in the syntax/concepts that were shown fast or left implicit.

---

## 0. Recap — Where We Are

Video 5 = first sequential/linear workflows in code. **This video = first
parallel workflows.** Two examples, increasing difficulty:
1. Non-LLM parallel workflow (cricket stats) — pure syntax, no LLM noise.
2. LLM-based parallel workflow (UPSC essay evaluator) — combines parallel
   nodes + **structured output** (from the LangChain playlist) + **reducers**
   (from Video 4).

---

## 1. Example 1 — Cricket Batsman Stats (non-LLM parallel workflow)

**Inputs:** `runs`, `balls`, `fours`, `sixes` (all for one innings).
**Outputs (computed in parallel):**
- **Strike Rate** = `(runs / balls) * 100`
- **Boundary %** = `((fours×4 + sixes×6) / runs) * 100` — % of runs scored via boundaries
- **Balls per Boundary** = `balls / (fours + sixes)` — average balls between boundaries

All three are independent of each other — none needs another's output — so
they can run **in parallel**, then a `summary` node merges the three results
into a single text summary.

**Graph shape:**
```
START ─┬─> calculate_strike_rate   ─┐
       ├─> calculate_boundary_pct  ─┼─> summary ─> END
       └─> calculate_balls_per_boundary ─┘
```

### State
```python
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    boundary_percent: float
    balls_per_boundary: float
    summary: str
```

### Nodes
```python
def calculate_strike_rate(state: BatsmanState) -> BatsmanState:
    strike_rate = (state["runs"] / state["balls"]) * 100
    state["strike_rate"] = strike_rate
    return state

def calculate_balls_per_boundary(state: BatsmanState) -> BatsmanState:
    bpb = state["balls"] / (state["fours"] + state["sixes"])
    state["bpb"] = bpb
    return state

def calculate_boundary_percent(state: BatsmanState) -> BatsmanState:
    boundary_percent = ((state["fours"] * 4 + state["sixes"] * 6) / state["runs"]) * 100
    state["boundary_percent"] = boundary_percent
    return state

def summary(state: BatsmanState) -> BatsmanState:
    summary_text = (
        f"Strike Rate: {state['strike_rate']} \n"
        f"Balls per Boundary: {state['bpb']} \n"
        f"Boundary Percent: {state['boundary_percent']}"
    )
    state["summary"] = summary_text
    return state
```

### Graph — fan-out then fan-in
```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(BatsmanState)
graph.add_node("calculate_strike_rate", calculate_strike_rate)
graph.add_node("calculate_balls_per_boundary", calculate_balls_per_boundary)
graph.add_node("calculate_boundary_percent", calculate_boundary_percent)
graph.add_node("summary", summary)

# fan-out: three parallel edges from START
graph.add_edge(START, "calculate_strike_rate")
graph.add_edge(START, "calculate_balls_per_boundary")
graph.add_edge(START, "calculate_boundary_percent")

# fan-in: all three converge on "summary"
graph.add_edge("calculate_strike_rate", "summary")
graph.add_edge("calculate_balls_per_boundary", "summary")
graph.add_edge("calculate_boundary_percent", "summary")

graph.add_edge("summary", END)

workflow = graph.compile()
```

### ⚠️ The Error You WILL Hit: `InvalidUpdateError`

Running the code above **as written** (each node returning the entire
`state` object) throws:
```
InvalidUpdateError: At key 'runs': Can receive only one value per step
```

**Why this happens:** each of the three parallel nodes returns the **full
state dict** — including `runs`, `balls`, `fours`, `sixes`, even though none
of them actually changed those fields. LangGraph can't tell "returned
unchanged" apart from "returned with an update," so when three nodes running
in the *same superstep* all "return" a value for the same key (`runs`), it
looks like three conflicting simultaneous writes to one field — which
isn't allowed without a reducer telling LangGraph how to combine them.

**The fix — return only a *partial* update (a plain dict with just the
changed keys), not the whole state:**
```python
def calculate_strike_rate(state: BatsmanState) -> dict:
    strike_rate = (state["runs"] / state["balls"]) * 100
    return {"strike_rate": strike_rate}     # only this key, not the full state

def calculate_balls_per_boundary(state: BatsmanState) -> dict:
    bpb = state["balls"] / (state["fours"] + state["sixes"])
    return {"bpb": bpb}

def calculate_boundary_percent(state: BatsmanState) -> dict:
    boundary_percent = ((state["fours"] * 4 + state["sixes"] * 6) / state["runs"]) * 100
    return {"boundary_percent": boundary_percent}

def summary(state: BatsmanState) -> dict:
    summary_text = (
        f"Strike Rate: {state['strike_rate']} \n"
        f"Balls per Boundary: {state['bpb']} \n"
        f"Boundary Percent: {state['boundary_percent']}"
    )
    return {"summary": summary_text}
```

> 🔧 Gap-fill — a subtlety worth being precise about, since the video's
> phrasing ("nodes expect and return a dictionary, not literally a State
> object") is a bit loose: a node's return value is **merged** into the
> shared state, key by key. A node **always CAN** return the full state
> dict safely *when it's the only writer for every key in it* (true in a
> purely sequential graph — see Video 5). The rule that actually matters:
> **never let two nodes that can execute in the same superstep both "touch"
> the same key unless that key has a reducer.** Returning the full state
> from a parallel node touches every key in it, including untouched ones —
> that's the real problem, not "returning full state" in general.

**Rule of thumb going forward (works for both sequential AND parallel):**
> Always return a **partial update** (only the keys a node actually
> changes) — the teacher explicitly recommends adopting this as a universal
> habit, not just a parallel-workflow-only fix, since it works everywhere.

### Run it
```python
initial_state = {"runs": 100, "balls": 50, "fours": 6, "sixes": 4}
final_state = workflow.invoke(initial_state)
print(final_state)
```

> 🔧 Gap-fill — the video also live-debugs a **formula bug**, worth noting
> since it's an easy mistake: strike rate should be
> `(runs / balls) * 100` (multiply by 100), not divide by 100. Watch for
> this when computing any "per-100" rate metric.

---

## 2. Example 2 — UPSC Essay Evaluator (LLM-based parallel workflow)

Extends the essay-evaluation example from Video 4 into working code. This
example deliberately combines **three concepts at once**:
1. Parallel workflows (this video)
2. **Structured Output** (carried over from the LangChain playlist)
3. **Reducers** (from Video 4)

**Workflow:**
```
START ─┬─> evaluate_language   (LLM call #1) ─┐
       ├─> evaluate_analysis   (LLM call #2) ─┼─> final_evaluation ─> END
       └─> evaluate_thought    (LLM call #3) ─┘
```
Each of the three evaluator nodes independently sends the essay to an LLM
and asks it to judge one dimension (**language quality**, **depth of
analysis**, **clarity of thought**), returning both a **text feedback** and
an **integer score (0–10)** for that dimension. `final_evaluation` merges
the three feedback texts into one summarized feedback (via another LLM
call) and averages the three scores into a final score.

### Why Structured Output is required here

> 🔧 Gap-fill — this is the crux insight, stated but not deeply justified in
> the video: if you just prompt the LLM in plain English ("give me feedback
> and a score"), the LLM might reliably follow the format ~8/10 times, but
> occasionally return the score as text ("seven") instead of a number —
> which breaks any downstream math (like averaging). **Structured output
> forces the model to return machine-parseable, schema-conformant JSON every
> time**, which matters here specifically because the final node does
> arithmetic (`average`) on the scores.

**Requires a model that supports structured output well** (e.g. `gpt-4o-mini`).

**Define the schema with Pydantic:**
```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

model = ChatOpenAI(model="gpt-4o-mini")
structured_model = model.with_structured_output(EvaluationSchema)
```

> 🔧 Gap-fill — `ge=0, le=10` (Pydantic's "greater than or equal" /
> "less than or equal") are the actual validation constraints the video
> refers to as "rules that the score should be between 0 and 10" without
> naming them — Pydantic will reject/retry on out-of-range values.

**Test it standalone before wiring into a graph** (good practice worth
copying in your own work):
```python
prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10:\n{essay}"
result = structured_model.invoke(prompt)
print(result.score)      # e.g. 8
print(result.feedback)   # detailed text
```
> Calling `model.invoke(...)` (the *un*-structured model) here returns a
> plain `AIMessage`; you must call `structured_model.invoke(...)` to get
> back an `EvaluationSchema` object with `.score` / `.feedback` attributes.

### State — where the Reducer comes in
```python
from typing import TypedDict, Annotated
from operator import add

class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], add]   # <-- reducer!
    avg_score: float
```

**Why `individual_scores` needs a reducer:** the three evaluator nodes run
**in parallel**, and each one wants to contribute **one score** into the
**same** `individual_scores` list. Without a reducer, this is exactly the
`InvalidUpdateError` situation from Example 1 — three simultaneous writers
to one key. Here, though, we don't want to *overwrite* the list — we want
**all three scores preserved**. So the fix isn't "return a partial update"
(that alone doesn't merge lists); it's telling LangGraph explicitly **how**
to combine multiple writes to this key.

> 🔧 Gap-fill — walking through exactly what happens, since the video's
> explanation is a bit hand-wavy about the mechanics:
> 1. Each node returns `{"individual_scores": [8]}` — a **single-item list**,
>    not a bare integer. (This is the part easy to miss — you must wrap the
>    score in a list, because the reducer combines *lists*, matching the
>    field's declared type `list[int]`.)
> 2. LangGraph sees three concurrent partial updates to `individual_scores`:
>    `[8]`, `[7]`, `[6]`.
> 3. Because the field is `Annotated[list[int], add]`, LangGraph applies
>    Python's `operator.add` (which is just `+` as a function) to combine
>    them: `[8] + [7] + [6] = [8, 7, 6]`.
> 4. `operator.add` is used (instead of the literal `+` operator) because
>    you can't write `+` as a bare reference in a type annotation — you need
>    a **callable**, and `operator.add` is the functional/importable version
>    of the `+` operator, living in Python's built-in `operator` module.

> Other reducers exist too — e.g. `max`, `min` — used the same way
> (`Annotated[int, max]`) when you want "keep the highest value seen" rather
> than "combine all values" behavior. Pick whichever combining rule fits the
> field's semantics.

### Nodes
```python
def evaluate_language(state: UPSCState) -> dict:
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "language_feedback": output.feedback,
        "individual_scores": [output.score],
    }

def evaluate_analysis(state: UPSCState) -> dict:
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "analysis_feedback": output.feedback,
        "individual_scores": [output.score],
    }

def evaluate_thought(state: UPSCState) -> dict:
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10:\n{state['essay']}"
    output = structured_model.invoke(prompt)
    return {
        "clarity_feedback": output.feedback,
        "individual_scores": [output.score],
    }

def final_evaluation(state: UPSCState) -> dict:
    # Text summary uses the PLAIN model, not the structured one —
    # we don't want it forced to also emit a score field here.
    prompt = (
        "Based on the following feedbacks, create a summarized feedback:\n"
        f"Language feedback: {state['language_feedback']}\n"
        f"Depth of analysis feedback: {state['analysis_feedback']}\n"
        f"Clarity of thought feedback: {state['clarity_feedback']}"
    )
    overall_feedback = model.invoke(prompt).content

    avg_score = sum(state["individual_scores"]) / len(state["individual_scores"])

    return {
        "overall_feedback": overall_feedback,
        "avg_score": avg_score,
    }
```

> 🔧 Gap-fill — important, easy-to-miss design choice the video makes but
> doesn't explicitly flag: `final_evaluation` uses the **unstructured**
> `model`, not `structured_model`. If you used the structured model here, it
> would be forced to also emit a `score` field (per `EvaluationSchema`),
> which isn't what you want for a pure text-summarization step — you're
> already computing `avg_score` yourself from `individual_scores`.

### Graph
```python
graph = StateGraph(UPSCState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()
```

### Run it
```python
initial_state = {"essay": essay}   # essay = a sample string
final_state = workflow.invoke(initial_state)
print(final_state)
# individual_scores: [7, 8, 8], avg_score: 7.67, overall_feedback: "...", etc.
```

---

## 3. Concept Map — What This Video Actually Combined

| Concept | Where it's from | Role here |
|---|---|---|
| Parallel nodes (fan-out/fan-in edges) | This video | Run 3 evaluations concurrently instead of sequentially |
| Partial state updates | This video (Example 1) | Avoid `InvalidUpdateError` when multiple nodes touch state simultaneously |
| Structured Output (Pydantic schema + `with_structured_output`) | LangChain playlist (referenced, not re-taught) | Force reliable, parseable `{feedback, score}` JSON every call |
| Reducers (`Annotated[list, add]`) | Video 4 | Merge 3 parallel single-item score lists into one combined list, instead of one overwriting another |

---

## 4. Self-Test (revision)

1. What error do you get if a parallel node returns the entire state instead of a partial update? *(`InvalidUpdateError` — "can receive only one value per step")*
2. Why doesn't a plain partial-update dict fix the `individual_scores` problem in Example 2? *(a partial update alone still replaces the key rather than merging lists — you need a reducer to combine multiple writes into one list)*
3. Why is the score wrapped as `[output.score]` (a list) instead of returned as a bare int? *(the reducer `operator.add` combines lists via `+`; the field type is `list[int]`, so each contribution must already be a list)*
4. Why does `evaluate_language`/`evaluate_analysis`/`evaluate_thought` need structured output but `final_evaluation`'s feedback-summarizing step doesn't? *(the per-dimension nodes must return a reliable numeric score for later averaging; the summarizer is pure text generation with no numeric output required)*
5. What Python module/function is `operator.add` used for here, and why not just `+`? *(it's the functional/importable equivalent of `+`; type annotations need a callable reference, not a bare operator symbol)*
6. What's the universal rule of thumb the teacher recommends for return values from any node, sequential or parallel? *(always return a partial update — only the keys actually changed — rather than the full state)*