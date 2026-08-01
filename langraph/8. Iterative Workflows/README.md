# Video 8: Iterative (Looping) Workflows in LangGraph

**Playlist:** Agentic AI using LangGraph (CampusX, Nitish)
**Video topic:** 4th type of workflow — Iterative/Looping Workflows. Practical example: Twitter/X post generator with Generator → Evaluator → Optimizer loop.

---

## 1. Recap: 4 Types of Workflows in LangGraph

| # | Workflow type | Kya hota hai |
|---|---|---|
| 1 | **Sequential** | Tasks linearly ek ke baad ek execute |
| 2 | **Parallel** | Multiple tasks ek saath execute, phir merge |
| 3 | **Conditional** | Branches available, but sirf ek condition ke basis pe select hota hai |
| 4 | **Iterative / Looping** *(is video ka topic)* | Do (ya zyada) nodes ke beech **loop** chalta hai jab tak koi cheez improve na ho jaaye ya ek limit na aa jaaye |

> **Core idea:** Iterative workflow tab use hota hai jab aapko kisi output ko **repeatedly refine/improve** karna ho — jaise ek draft ko baar-baar better banate jaana jab tak wo "good enough" na ho jaaye.

---

## 2. Real-World Use Case: Auto Social Media Post Generator

**Problem:** Content creator (jaise YouTube pe active, lekin LinkedIn/X/Instagram pe kam active) chahta hai ek **automated workflow** jo topic diye jaane pe achha post generate kare aur post kare. Lekin problem: LLM se **first-attempt output** usually satisfactory nahi hota.

**Solution:** Iterative workflow — jahan post baar-baar generate + evaluate + optimize hota hai jab tak quality bar cross na ho jaaye.

### 2.1 Three Core Components
| Component | Role |
|---|---|
| **Generator** | Topic leke ek post/tweet generate karta hai |
| **Evaluator** | Strict criteria ke against post ko judge karta hai → `Approved` ya `Needs Improvement` + feedback |
| **Optimizer** | Evaluator ka feedback leke post ko improve karta hai |

### 2.2 Workflow Diagram
```
START → generate → evaluate ─┬─(Approved)──→ END (or Human-in-loop → post via API)
                              └─(Needs Improvement)──→ optimize ──→ evaluate (LOOP BACK)
```

> **Real-world note (video mein explicitly bola gaya):** Production mein "Approved" ke baad seedha END nahi hota — waha **human-in-the-loop** checkpoint aata hai jahan human final approval deta hai, tab actual API call se post publish hoti hai. Ye Video 2 ke "Supervisor" component se directly connect karta hai.

---

## 3. Why Different LLMs for Different Roles?

Video mein ek important design principle bataya gaya: **ideally** teeno roles (Generator, Evaluator, Optimizer) ke liye **alag-alag specialized LLMs** use karne chahiye:
- **Generator** → strong writing capability wala model (jaise GPT-4.5)
- **Evaluator** → jo instructions ko strictly/religiously follow kare
- **Optimizer** → phir se strong rewriting capability

> **Gap-fill note:** Ye ek important production insight hai jo aage LangGraph practical work mein bahut kaam aayega — **"model selection per role"**, na ki sirf ek hi LLM sab kaam ke liye use karna. Different LLM providers/models ke alag-alag strengths hote hain:
> - Kuch models creative writing mein better hote hain (jaise Claude, GPT-4.5)
> - Kuch models **instruction-following aur structured evaluation** mein better/consistent hote hain
> - Chhote/cheaper models (jaise GPT-4o-mini) **evaluator** role ke liye kaafi ho sakte hain, jisse cost bhi bache
>
> Video mein demo ke liye simplicity ke wajah se roughly same models use hue (dummy project), lekin real project mein ye choice explicitly research karke karni chahiye.

---

## 4. Code Walkthrough

### 4.1 Three LLM instances
```python
generator_llm = ChatOpenAI(model="gpt-4o")
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
optimizer_llm = ChatOpenAI(model="gpt-4o")
```

### 4.2 State Definition
```python
from typing import Literal, Annotated
from typing_extensions import TypedDict
import operator

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
```

| Field | Purpose |
|---|---|
| `topic` | User-given topic |
| `tweet` | Currently generated tweet |
| `evaluation` | `"approved"` or `"needs_improvement"` — Literal type se restrict kiya gaya |
| `feedback` | Evaluator ka constructive feedback |
| `iteration` | Kitne loops ho chuke hain (loop-breaking ke liye) |
| `max_iteration` | Loop ki upper limit (video mein 5 use hua) |
| `tweet_history` / `feedback_history` | Har iteration ke tweet/feedback ka **accumulated list** |

> **Critical concept — Reducer Function (`operator.add`):**
> Normally LangGraph mein jab ek node state ka koi key update karta hai, wo **replace** ho jaata hai purani value ko. Lekin `tweet_history` jaisi list mein hume chahiye ki har naya tweet **purano ke saath merge/append** ho, replace na ho.
>
> `Annotated[list[str], operator.add]` — ye syntax LangGraph ko batata hai: "jab bhi is field ke liye koi node ek naya value return kare, use `operator.add` (i.e. list concatenation, `+`) function se **combine** karo purani value ke saath, replace mat karo."
>
> Isliye node ke andar likhte waqt: `"tweet_history": [response]` (list ke andar single value) — kyunki `operator.add` do lists ko concatenate karta hai (`old_list + [response]`), single string ko nahi.

### 4.3 Loop-breaking safeguard — `max_iteration`
> **Why this matters:** Agar Evaluator baar-baar reject karta rahe aur Optimizer improve na kar paaye (ya evaluation criteria bahut strict ho, ya LLM itna capable na ho), to ye **infinite loop** ban sakta hai. `max_iteration` ek **hard safety limit** hai jo guarantee karta hai ki loop kabhi na kabhi terminate ho.
>
> Ye ek **universal pattern** hai kisi bhi agentic/iterative system mein — har loop ko ek explicit exit condition chahiye, sirf "success" pe nahi, balki "max attempts reached" pe bhi.

### 4.4 Generator Node
```python
def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter influencer."),
        HumanMessage(content=f"""
            Write a short, original, and hilarious tweet on the topic: {state['topic']}
            Rules:
            - Do not use question-answer format
            - Max 280 characters
            - Use observational humor, irony, sarcasm, cultural references
            - Think in meme logic, punchlines, and relatable text
            - Use simple, day to day English
        """)
    ]
    response = generator_llm.invoke(messages).content
    return {
        "tweet": response,
        "tweet_history": [response]
    }
```

> **Gap-fill note — SystemMessage/HumanMessage vs plain f-string prompts:** Is video mein pehli baar (playlist mein) explicit `SystemMessage` + `HumanMessage` objects use hue (list ke andar), instead of ek plain f-string. Ye industry-standard **chat message format** hai jo har LLM provider (OpenAI, Anthropic, etc.) support karta hai:
> - `SystemMessage` — model ki "persona"/role/behavior define karta hai (high-level instructions)
> - `HumanMessage` — actual user query/task
> - (Ek teesra bhi hota hai — `AIMessage` — jo model ke pichle responses represent karta hai, multi-turn conversations mein)
>
> Ye separation important hai kyunki system message aur human message ko LLM slightly differently weigh karta hai — system message zyada "persistent instruction" ki tarah treat hota hai.

### 4.5 Evaluator Node (with Structured Output)
```python
class TweetEvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Final evaluation result"
    )
    feedback: str = Field(
        description="Constructive feedback for the tweet"
    )

structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluationSchema)

def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless, no-laugh-given Twitter critic. You evaluate tweets based on humor, originality, virality, and tweet format."),
        HumanMessage(content=f"""
            Evaluate the following tweet: {state['tweet']}

            Criteria:
            1. Originality — is it fresh, not overused/seen before
            2. Humor — genuinely funny?
            3. Punchiness — short and impactful?
            4. Virality Potential — shareable?
            5. Format — no Q&A style, under 280 characters, no traditional/cliché jokes

            Auto-reject if: Q&A format, over 280 characters, or traditional/cliché joke format.

            Respond only in structured format: evaluation (approved/needs_improvement) and feedback.
        """)
    ]
    response = structured_evaluator_llm.invoke(messages)
    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback,
        "feedback_history": [response.feedback]
    }
```

> **Key insight — evaluation criteria ki quality = output ki quality:** Video mein bar-bar emphasize hua ki jitna detailed/specific evaluation criteria hoga (originality, humor, punchiness, virality, format — with explicit auto-reject rules), utna hi better final output milega. Ye **prompt engineering** ka ek core principle hai jo agentic evaluator design mein especially critical hai — vague criteria ("is this good?") se vague/inconsistent evaluation milta hai.

### 4.6 Optimizer Node
```python
def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You punch up tweets for virality and humor based on given feedback."),
        HumanMessage(content=f"""
            Improve the tweet based on this feedback: {state['feedback']}
            Topic of the tweet: {state['topic']}
            Original tweet: {state['tweet']}

            Rewrite it as a short, viral-worthy tweet. Avoid Q&A style and stay under 280 characters.
        """)
    ]
    response = optimizer_llm.invoke(messages).content
    iteration = state['iteration'] + 1
    return {
        "tweet": response,
        "iteration": iteration,
        "tweet_history": [response]
    }
```

Note: `iteration` yahan **manually increment** ho raha hai (`state['iteration'] + 1`) — LangGraph khud automatically iteration count track nahi karta, ye developer ki responsibility hai.

### 4.7 Router Function for the Loop
```python
def route_evaluation(state: TweetState) -> str:
    if state['evaluation'] == 'approved' or state['iteration'] >= state['max_iteration']:
        return 'approved'
    else:
        return 'needs_improvement'
```

> **Important detail:** Ye router **do conditions ko OR se combine** karta hai — `approved` (quality criteria met) YA `max_iteration` reach ho gaya (safety limit). Dono cases mein loop terminate hona chahiye, chahe quality achieve na hui ho. Ye graceful degradation ka example hai — infinite loop se better hai ek "best effort" output dena.

### 4.8 Building the Graph (Edges — this is where the LOOP happens)
```python
graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges("evaluate", route_evaluation, {
    "approved": END,
    "needs_improvement": "optimize"
})

graph.add_edge("optimize", "evaluate")   # <-- THIS edge creates the loop

workflow = graph.compile()
```

> **The entire "loop" concept, boiled down:** Loop banane ke liye koi special LangGraph feature nahi chahiye — sirf ek **normal edge jo waapas peeche kisi pehle node ki taraf point kare** (`optimize → evaluate`). Jab tak conditional edge kisi na kisi condition pe `END` na bheje, ye cycle chalta rehta hai. Ye video ka sabse important one-line takeaway hai: **"loops = edges jo cyclically wapas jaate hain, combined with ek conditional exit."**

---

## 5. Sample Run Observations (from video)

- Topic: "Indian Railways", `iteration: 1`, `max_iteration: 5`
- Kai baar first attempt hi approve ho gaya (ek hi iteration mein)
- Ek baar mein GPT-4o (strong model) use kiya to bahut fast approve ho gaya — kam iterations dikhe demo ke liye
- Instructor ne **GPT-4o-mini** (weaker model) generator/optimizer ke liye use kiya taaki weak tweet generate ho aur evaluator use reject kare — isse loop 2-3 iterations tak chala aur demonstration behtar hua

> **Practical lesson (not explicitly stated but implied):** Stronger models kam iterations mein achha output de sakte hain (cost-effective long-term), lekin demo/testing ke liye weaker model use karna loop-behavior verify karne ke liye useful hota hai.

---

## 6. Gaps Filled / Additional Context

### 6.1 Reducer functions — broader concept
Video mein `operator.add` reducer use hua tha "last video" (Parallel Workflows) mein bhi (jaisa instructor ne reference diya). Ye ek broader LangGraph concept hai:

| Reducer | Behavior |
|---|---|
| Default (no reducer) | Naya value purane ko **replace** karta hai |
| `operator.add` | Naya value purane mein **append/concatenate** hota hai (lists ke liye `+`, strings ke liye concatenation) |
| Custom reducer function | Aap khud ka function likh sakte ho jo custom merge-logic define kare (jaise dictionary merge, dedup logic, etc.) |

Ye pattern especially useful hota hai jab **parallel branches** same key ko update kar rahe hon (jahan simple replace se data loss ho jaata), ya jaise is video mein — jab ek key ko **multiple times, sequentially** update karna ho without losing history.

### 6.2 Connecting to earlier videos
- **Video 2's "Planning is iterative"** — waha bola gaya tha ki planning aur execution ek loop mein chalte hain jab tak goal achieve na ho. Ye video exactly wahi concept practically implement karta hai.
- **Video 2's "Reasoning in error handling"** — Evaluator ka reject karna aur Optimizer ka fix karna, ye conceptually reasoning-driven adaptive behavior hai (Video 2 ke "Adaptability" characteristic se match karta hai).
- **Video 7's Conditional Workflow (`add_conditional_edges`)** — is video ka loop bhi conditional edges hi use karta hai; naya sirf ye hai ki ek branch (`optimize`) **wapas peeche** point kar rahi hai, jisse cycle banta hai.

### 6.3 This pattern has a well-known name in the industry
Ye exact "Generator → Evaluator → Optimizer" loop pattern LLM engineering community mein **"Generator-Critic" pattern** ya **"Self-Refine" pattern** ke naam se jaana jaata hai. Kai research papers (jaise *Self-Refine: Iterative Refinement with Self-Feedback*) isi structure ko formalize karte hain — LLM khud apne output ko critique karke improve karta hai. LangGraph mein ye ek official example/template ke roop mein bhi documented hai ("Reflection" pattern ke naam se).

> Agar interview mein "agentic design patterns" pucha jaaye, to "Reflection" / "Generator-Critic" is video ka topic hai — worth remembering by name.

### 6.4 Human-in-the-loop ka missing piece
Video ne explicitly bola ki real production mein `approved` ke baad seedha END nahi, balki human approval + actual API-call-to-post-on-platform aana chahiye. Ye is video mein implement nahi hua (future video ka topic — jab "Tools" aur "Human-in-the-loop" cover honge). Iska matlab hai ye current workflow ek **incomplete/demo pipeline** hai jo aage extend hoga:
```
... → evaluate (approved) → [human review] → [API call to post] → END
```

---

## 7. Quick Revision Table

| Concept | One-liner |
|---|---|
| Iterative/Looping Workflow | Do nodes ke beech baar-baar chalna jab tak improve/limit na ho jaaye |
| Generator-Evaluator-Optimizer | Ek common 3-role pattern for iterative refinement (aka "Reflection"/"Self-Refine" pattern) |
| `Annotated[list, operator.add]` | Reducer — naya value replace nahi, purane mein **append** hota hai |
| `max_iteration` safeguard | Infinite loop rokne ke liye hard exit condition |
| Loop in LangGraph | Sirf ek edge jo **wapas peeche** kisi node ki taraf jaaye, + conditional exit |
| SystemMessage / HumanMessage | Standard chat message format — persona vs actual task instructions |
| Model selection per role | Alag tasks (generate/evaluate/optimize) ke liye alag LLMs choose karna better hota hai |