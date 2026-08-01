# Video 7: Conditional Workflows in LangGraph

**Playlist:** Agentic AI using LangGraph (CampusX, Nitish)
**Video topic:** Conditional Workflows — `add_conditional_edges` ka use karke branching logic implement karna (2 practical examples: Quadratic Equation Solver + Customer Review Responder)

> **Note:** Ye playlist ka 7th video hai. Isse pehle ke videos (3-6) mein probably **Sequential Workflows** aur **Parallel Workflows** cover hue honge (jaise video mein reference diya gaya hai) — agar wo transcripts bhi share karoge to unke notes bhi bana dunga taaki poori playlist ka reference doc continuous rahe.

---

## 1. Recap: 3 Types of Workflows in LangGraph

| Workflow type | Kya hota hai |
|---|---|
| **Sequential** | Tasks ek ke baad ek linearly execute hote hain |
| **Parallel** | Multiple tasks **same time** pe execute hote hain (branch karke sab paths mein ek saath jaana), phir results merge hote hain |
| **Conditional** *(is video ka topic)* | Multiple branches available hote hain, lekin **ek condition ke basis pe sirf ek hi branch** choose hoti hai — baaki skip |

> **Key distinction (Parallel vs Conditional):** Dono mein branches dikhte hain graph mein, lekin:
> - Parallel: saare branches ek saath execute hote hain
> - Conditional: sirf **ek** branch execute hota hai, jaise programming mein `if-else`
>
> Conditional workflow ka core mental model: **"if-else, but for workflows."** Jitna important if-else normal programming mein hai, utna hi important conditional branching LangGraph mein hai — aage complex workflows mein ye almost hamesha use hoga.

---

## 2. Example 1: Quadratic Equation Solver (Non-LLM Workflow)

### 2.1 Problem Recap (Maths)
Quadratic equation: `ax² + bx + c = 0`

**Discriminant:** `D = b² - 4ac`

| Condition | Result |
|---|---|
| D > 0 | 2 distinct real roots |
| D = 0 | 1 repeated root |
| D < 0 | No real roots |

**Formulas:**
- Two real roots: `x = (-b ± √D) / 2a`
- Repeated root: `x = -b / 2a`

### 2.2 Workflow Diagram
```
START → show_equation → calculate_discriminant → [CONDITIONAL BRANCH]
                                                        ├── real_roots       ─┐
                                                        ├── repeated_roots   ─┼─→ END
                                                        └── no_real_roots    ─┘
```

### 2.3 Code Walkthrough

**Step 1 — Define State**
```python
from typing_extensions import TypedDict

class QuadState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str
```

**Step 2 — Basic nodes (show_equation, calculate_discriminant)**
```python
def show_equation(state: QuadState):
    equation = f"{state['a']}x² + {state['b']}x + {state['c']}"
    return {"equation": equation}

def calculate_discriminant(state: QuadState):
    discriminant = (state['b'] ** 2) - (4 * state['a'] * state['c'])
    return {"discriminant": discriminant}
```

**Step 3 — Conditional branch nodes**
```python
def real_roots(state: QuadState):
    root1 = (-state['b'] + state['discriminant'] ** 0.5) / (2 * state['a'])
    root2 = (-state['b'] - state['discriminant'] ** 0.5) / (2 * state['a'])
    result = f"The roots are {root1} and {root2}"
    return {"result": result}

def repeated_roots(state: QuadState):
    root = -state['b'] / (2 * state['a'])
    result = f"Only repeating root is {root}"
    return {"result": result}

def no_real_roots(state: QuadState):
    result = "No real roots"
    return {"result": result}
```

**Step 4 — Router function (condition check)**
```python
def check_condition(state: QuadState) -> str:
    if state['discriminant'] > 0:
        return "real_roots"
    elif state['discriminant'] == 0:
        return "repeated_roots"
    else:
        return "no_real_roots"
```

> **Important pattern:** Router function ka naam graph node ka naam nahi hota — ye ek **separate helper function** hai jiska sirf ek kaam hai: state dekh ke agle node ka **naam (string)** return karna.

**Step 5 — Build graph**
```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(QuadState)

graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_roots", repeated_roots)
graph.add_node("no_real_roots", no_real_roots)

graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")

# CONDITIONAL EDGE — ye hi is video ka core concept hai
graph.add_conditional_edges("calculate_discriminant", check_condition)

graph.add_edge("real_roots", END)
graph.add_edge("repeated_roots", END)
graph.add_edge("no_real_roots", END)

workflow = graph.compile()
```

### 2.4 `add_conditional_edges` — Core Syntax
```python
graph.add_conditional_edges(<source_node>, <router_function>)
```
- `source_node` — kahan se branching start ho rahi hai
- `router_function` — ye function state ko dekh ke **next node ka naam (string)** return karta hai; graph ke andar automatically wahi node target ban jaata hai

> **Gap-fill note — video mein thoda implicit tha:** `add_conditional_edges` ke actual signature mein ek teesra optional parameter bhi hota hai — a **mapping dictionary** jo router function ke return values ko explicit node names se map karta hai:
> ```python
> graph.add_conditional_edges(
>     "calculate_discriminant",
>     check_condition,
>     {
>         "real_roots": "real_roots",
>         "repeated_roots": "repeated_roots",
>         "no_real_roots": "no_real_roots",
>     }
> )
> ```
> Video mein direct form use hua (jahan router return values khud hi node names ke exactly same the), isliye mapping dictionary implicit rehti hai. Lekin agar aapke router function ke return values node names se **match nahi karte** (jaise router "yes"/"no" return kare lekin nodes "path_a"/"path_b" naam ke hon), tab ye mapping dictionary explicitly deni padegi.

### 2.5 Visual cue: Dotted vs Solid Edges
Graph visualization mein:
- **Solid edges** = normal/fixed edges (`add_edge`)
- **Dotted edges** = conditional edges (`add_conditional_edges`) — indicate karta hai ki in edges mein se **runtime pe sirf ek** choose hoga

---

## 3. Example 2: LLM-based Customer Review Responder

### 3.1 Problem Statement
Customer review aata hai → sentiment detect karo → agar **positive** hai to seedha thank-you reply likho; agar **negative** hai to pehle diagnosis (issue type, tone, urgency) karo, phir uske basis pe empathetic reply likho.

### 3.2 Workflow Diagram
```
START → find_sentiment → [CONDITIONAL BRANCH]
                              ├── positive → positive_response → END
                              └── negative → run_diagnosis → negative_response → END
```

### 3.3 Structured Output — Core Concept
Jab LLM se aapko **free-form text** nahi, balki ek **fixed structure** (jaise sirf "positive" ya "negative") chahiye, tab **structured output** use karte hain.

**Step 1 — Define schema using Pydantic**
```python
from pydantic import BaseModel, Field
from typing import Literal

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )

structured_model = model.with_structured_output(SentimentSchema)
```

- `Literal["positive", "negative"]` — LLM ke output ko sirf in 2 values tak restrict karta hai
- `model.with_structured_output(schema)` — ek naya model return karta hai jiska output hamesha is schema ke format mein aayega (JSON internally, Pydantic object externally)

**Usage:**
```python
result = structured_model.invoke("What is the sentiment of: The software is too bad")
result.sentiment  # → "negative"
```

> **Gap-fill note — `with_structured_output` internals:** Video ne isko as-is use kiya bina explain kiye ki backend mein kya ho raha hai. Ye actually LLM provider ke **function calling / JSON mode** capability ko use karta hai — LangChain internally aapke Pydantic schema ko ek JSON schema mein convert karta hai aur model ko "function call" ke roop mein present karta hai, jisse model ka output guaranteed structurally valid rehta hai (normal text-parsing se zyada reliable, kyunki aapko manually JSON parse + validate nahi karna padta). Ye same concept hai jo Video 2 ke "Tool Selection" (Brain component) discussion se connect karta hai.

### 3.4 Second Schema — Diagnosis
```python
class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(
        description="The category of issue mentioned in the review"
    )
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(
        description="The emotional tone expressed by the user"
    )
    urgency: Literal["low", "medium", "high"] = Field(
        description="How urgent or critical the issue appears to be"
    )

structured_model_2 = model.with_structured_output(DiagnosisSchema)
```

> Yaha 3 alag-alag Literal fields ek hi schema mein combine ho rahe hain — ye demonstrate karta hai ki structured output sirf single-field tak limited nahi hai; complex multi-field JSON bhi ek hi LLM call mein extract ho sakta hai.

### 3.5 Full State Definition
```python
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str
```

> **Note:** `diagnosis` yahan ek **nested dictionary** hai — state ke andar state jaisa. Ye pattern useful hai jab ek node ka output multiple related fields ka group ho (yahan issue_type + tone + urgency).

### 3.6 Node Functions

```python
def find_sentiment(state: ReviewState):
    prompt = f"For the following review find out the sentiment:\n{state['review']}"
    sentiment = structured_model.invoke(prompt).sentiment
    return {"sentiment": sentiment}

def check_sentiment(state: ReviewState) -> str:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank-you message in response to this review:
    \n{state['review']}\n
    Also, kindly ask the user to leave feedback on our website."""
    response = model.invoke(prompt).content
    return {"response": response}

def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:\n{state['review']}
    Return issue_type, tone and urgency."""
    response = structured_model_2.invoke(prompt)
    return {"diagnosis": response.model_dump()}

def negative_response(state: ReviewState):
    diagnosis = state['diagnosis']
    prompt = f"""You are a support assistant.
    The user had a '{diagnosis['issue_type']}' issue,
    sounded '{diagnosis['tone']}',
    and marked urgency as '{diagnosis['urgency']}'.
    Write an empathetic, helpful resolution message."""
    response = model.invoke(prompt).content
    return {"response": response}
```

> **Gap-fill note — `.model_dump()`:** Ye Pydantic v2 ka method hai jo ek Pydantic `BaseModel` object ko plain Python `dict` mein convert karta hai (Pydantic v1 mein isko `.dict()` bola jaata tha — naming change hui hai versions ke beech). Video mein `response.model_dump()` use hua kyunki `diagnosis` state field ek `dict` type ka hai, lekin `structured_model_2.invoke()` se seedha ek Pydantic object milta hai — isliye conversion zaroori hai.

### 3.7 Graph Construction
```python
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")

graph.add_conditional_edges("find_sentiment", check_sentiment)

graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()
```

### 3.8 Sample Run
- **Positive review** → sentiment: `positive` → direct thank-you message generate hota hai
- **Negative review** ("app keeps freezing on login") → sentiment: `negative` → diagnosis: `{issue_type: "Bug", tone: "frustrated", urgency: "high"}` → empathetic resolution message generate hota hai jo specifically bug + frustration + high urgency ko address karta hai

---

## 4. Core Takeaway (Formula)

```
Conditional Workflow banane ke 3 steps:
1. Ek router function likho jo state check kare aur next node ka NAME (string) return kare
2. graph.add_conditional_edges(source_node, router_function) call karo
3. Har possible destination node ko END (ya aage) se connect karo
```

---

## 5. Gaps Filled / Additional Context

### 5.1 Two ways to build conditional edges
Video mein explicitly mention hua ki conditional edges banane ka **ek aur tareeka bhi hai** — using a **`Command`** object — jo "Dynamic Workflows" wale video mein cover hoga. Preview ke liye:
- `add_conditional_edges` approach: **declarative** — pehle se sab possible destinations define karo, phir router function decide karta hai kisme jaana hai
- `Command` approach: **imperative** — node ke andar hi seedha decide kar sakte ho ki agla node kaunsa hoga aur state kya update karni hai, ek hi return statement mein (`Command(goto="node_name", update={...})`)

`Command` approach zyada flexible hoti hai complex/dynamic routing ke liye (jaise jab destination pehle se fixed set na ho), jabki `add_conditional_edges` cleaner hai jab destinations fixed aur limited hon (jaise is video ke dono examples).

### 5.2 `Literal` type ka significance
Python ke `typing.Literal` ka use dono examples mein baar-baar hua — ye sirf type-hinting ke liye nahi hai, balki **LLM ke output ko constrain karne** ke liye critical hai jab `with_structured_output` ke saath use kiya jaaye. Bina `Literal` ke agar aap sirf `str` use karte, LLM koi bhi arbitrary string return kar sakta tha (jaise "Positive" vs "positive" vs "Somewhat positive") jo downstream `if-else` logic ko break kar deta.

### 5.3 State ke andar nested structures
`diagnosis: dict` jaisa nested field state mein rakhna ek common LangGraph pattern hai jab ek node ka output multiple related values ka bundle ho. Alternative approach: `diagnosis` ko bhi khud ek separate `TypedDict`/`Pydantic` type define kar sakte the (jaise `DiagnosisState`) taaki type-safety aur bhi strict ho — video ne simplicity ke liye plain `dict` use kiya.

### 5.4 Real-world relevance — connecting back to Video 1 & 2
Ye conditional-workflow pattern exactly wahi **"Orchestrator"** component hai jo Video 2 mein discuss hua tha (task sequencing + **conditional routing**). Sentiment-based customer support routing bhi Video 1 ke "customer support" use-case se directly connect karta hai — waha conceptually bataya gaya tha, yahan practically implement ho raha hai.

### 5.5 Common gotcha noticed in video
Video mein khud instructor ko discriminant sign-handling mein ek chhota display bug mila (`f"{a}x² + {b}x + {c}"` negative `b`/`c` values ke saath "+  -5x" jaisa galat dikhata hai). Ye ek practical reminder hai: production code mein aise f-strings likhte waqt sign-handling explicitly karni chahiye, jaise:
```python
sign = "+" if state['b'] >= 0 else "-"
equation = f"{state['a']}x² {sign} {abs(state['b'])}x ..."
```

---

## 6. Quick Revision Table

| Concept | One-liner |
|---|---|
| Conditional Workflow | Multiple branches, sirf **ek** condition ke basis pe execute hota hai (if-else jaisa) |
| `add_conditional_edges(node, router_fn)` | Router function ka return value hi agla node decide karta hai |
| Router function | State input leta hai, **next node ka naam (string)** output karta hai |
| `with_structured_output(Schema)` | LLM output ko fixed Pydantic schema mein force karta hai |
| `Literal[...]` | Output values ko specific fixed set tak restrict karta hai |
| `.model_dump()` | Pydantic object → plain dict (Pydantic v2 syntax) |
| Dotted edges (visualization) | Conditional edges — runtime pe ek hi path choose hota hai |