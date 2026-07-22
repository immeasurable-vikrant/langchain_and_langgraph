# Video 2: What is Agentic AI? (Definition, Characteristics & Components)

**Playlist:** Agentic AI using LangGraph (CampusX, Nitish)
**Video topic:** Agentic AI ka formal study — definition, 6 core characteristics, aur 5 core components

---

## 1. Formal Definition

> **Agentic AI is a type of AI that can take up a task and goal from a user and then work towards completing it on its own with minimal human guidance. It plans, takes actions, adapts to changes, and seeks help only when necessary.**

Simple shabdon mein: Agentic AI ek software paradigm hai jahan aap system ko sirf ek **goal** dete ho, aur system khud us goal ko achieve karne ke liye planning + execution karta hai — human involvement minimum rehta hai.

### Reactive (Gen AI) vs Proactive (Agentic AI) — Goa Trip Example

| | Generative AI Chatbot | Agentic AI System |
|---|---|---|
| Approach | Har step pe aap alag-alag sawaal poochte ho ("best travel mode?", "best hotels?", "best places?") | Sirf ek goal do: "mujhe in dates mein Goa jaana hai" |
| Behavior | Reactive — sirf jo poocha wahi answer | Proactive — poora itinerary khud plan karke deta hai |

Yehi analogy directly HR-recruiter scenario pe bhi apply hoti hai (jo Video 1 mein detail se cover hua tha).

> **Recap note:** Ye wahi HR-recruiter-hiring-a-backend-engineer scenario hai jo Video 1 mein tha. Is video mein Nitish sir isko dobara use karte hain taaki characteristics/components ko concretely illustrate kar sakein.

---

## 2. Six Core Characteristics of Agentic AI

Koi bhi system agentic hai ya nahi, ye check karne ke liye ye **6 traits** dekhne hote hain:

1. Autonomy
2. Goal-Oriented
3. Planning
4. Reasoning
5. Adaptability
6. Context Awareness

---

### 2.1 Autonomy

**Definition:** Autonomy refers to an AI system's ability to make decisions and take actions on its own to achieve a given goal, without needing step-by-step human instructions.

Autonomy sirf ek jagah nahi dikhti — ye **multiple levels** pe implemented hoti hai:
- **Execution autonomy** — plan ke steps khud-ba-khud execute karna
- **Decision-making autonomy** — kitne candidates shortlist karne hain, kis basis pe — ye khud decide karna
- **Tool-usage autonomy** — kaunsa tool kab use karna hai, ye khud choose karna

#### Autonomy ko control karne ke 4 tareeke
Autonomy powerful hai, lekin **risky bhi** ho sakti hai (jaise: galat salary pe offer roll out karna, biased shortlisting, bina pooche paid ads chala dena). Isliye production systems mein autonomy control karni padti hai:

| Control mechanism | Kya karta hai |
|---|---|
| **Scope of permissions** | Kaunse tools/actions agent independently kar sakta hai, uski limit define karna |
| **Human-in-the-loop checkpoints** | Kuch risky steps se pehle mandatory human approval |
| **Override controls** | User kabhi bhi agent ko pause/stop/modify kar sake |
| **Guardrails & policies** | Hard rules/ethical boundaries (e.g., "weekends pe interviews mat schedule karo") |

> **Gap-fill note:** Video mein ye control mechanisms list ho gaye, lekin industry mein iske liye ek formal naam hai — **"agent autonomy levels"** ya **"autonomy slider."** Kuch frameworks (jaise LangGraph khud) explicitly `interrupt()` / checkpoint mechanisms provide karte hain jisse aap graph execution ko kisi bhi node pe pause karke human approval le sakte ho, phir resume kar sakte ho. Ye exactly wahi "human-in-the-loop" concept LangGraph mein implement karne ka native tareeka hai — aage practical mein ye dekhne ko milega.

---

### 2.2 Goal-Oriented

**Definition:** Being goal-oriented means the AI system operates with a persistent objective in mind and continuously directs its actions to achieve that objective, rather than just responding to isolated prompts.

- Goal **compass** ki tarah kaam karta hai autonomy ke liye — bina goal ke autonomy directionless hoti.
- Goals do tarah ke ho sakte hain:
  - **Independent goal** — "hire a backend engineer"
  - **Goal with constraints** — "hire a backend engineer **who is remote**, **from India**, **within $X budget**"

#### Goals memory mein kaise store hote hain (conceptual JSON)
```json
{
  "main_goal": "Hire a backend engineer",
  "constraints": {
    "experience": "2-4 years",
    "remote": true,
    "tech_stack": ["Python", "Django"]
  },
  "status": "active",
  "created_at": "...",
  "progress": {
    "jd_created": true,
    "posted_on": ["LinkedIn", "Naukri"],
    "applications_received": 8,
    "interviews_scheduled": 2,
    "hired": false,
    "onboarding_done": false
  }
}
```

- **Goals can be altered mid-way** — jaise agar 7 din tak koi apply na kare, HR decide kar sakta hai "chodo, freelancer hi rakh lete hain." Goal change hote hi poori planning aur execution reroute ho jaati hai.

> **Extra context:** Ye JSON representation exactly wahi structure hai jo LangGraph mein **"state"** ke roop mein represent hota hai (a `TypedDict` ya `Pydantic` model). Jab hum LangGraph practically padhenge, ye "goal + progress" object hi humara **graph state** banega jo har node ke beech pass hota hai.

---

### 2.3 Planning

Nitish sir ke according ye **sabse important trait** hai. Har agentic AI system do steps mein operate karta hai:

```
Goal → [Planning] → Plan → [Execution] → Sub-goals achieved → Goal achieved
              ↑___________________________|
              (agar step fail ho jaye, dobara planning — iterative loop)
```

**Definition:** Planning is the agent's ability to break down a high-level goal into a structured sequence of actions and sub-goals.

#### Planning ke 3 steps

**Step 1 — Generate multiple candidate plans**
Agent sirf ek plan nahi banata, **multiple alternate plans** generate karta hai. Jaise:
- Plan A: JD banao → LinkedIn/Naukri/Instahyre pe post karo
- Plan B: Internal referral chalao, ya hiring agency approach karo

**Step 2 — Evaluate all plans** (criteria):
| Criterion | Kya check hota hai |
|---|---|
| Efficiency | Kaunsa plan fastest execute hoga |
| Tool availability | Plan ke liye required tools available hain ya nahi (e.g., agar Google Search API hi nahi hai to wo plan reject) |
| Cost | Budget constraints ke andar fit hota hai kya |
| Risk | Failure ka chance kitna hai |
| Alignment with constraints | Jaise "remote hiring" constraint LinkedIn se better satisfy hota hai ya referral se |

**Step 3 — Select the best plan**
- **Human-in-the-loop input** lekar, ya
- **Pre-programmed policy** ke basis pe

> **Gap-fill note — this is literally "Tree of Thought" / search-based planning:** Video mein ye explicitly nahi bola gaya, lekin jo process describe hua hai ("initial state → final state, multiple paths exist, best path choose karo") ye exactly classical AI planning ka **search problem** hai jo Video 1 mein "traditional AI" discussion ke roots se connect hota hai. LLM-based agents mein isko formalize karne ke liye kuch known techniques hain:
> - **ReAct (Reason + Act)** — sabse common pattern, single-plan generate-and-execute-with-feedback loop
> - **Tree of Thoughts (ToT)** — explicitly multiple candidate plans generate karke unko evaluate/prune karna (yehi is video ka "Step 1 + Step 2" formalize karta hai)
> - **Plan-and-Execute** — pehle poora plan banao, phir execute karo (LangGraph ka ek standard example/template hai isi naam se)
>
> LangGraph mein "Plan-and-Execute" agent banane ka official pattern hai jisme ek "planner" node poore steps generate karta hai aur ek "executor" node unhe run karta hai — bilkul isi video ke structure jaisa.

---

### 2.4 Reasoning

**Definition:** Reasoning is the cognitive process through which an agentic AI system interprets information, draws conclusions, and then makes decisions — both while planning and executing.

**Human analogy:** Phone chori hone ka example — Environment se info mili (phone missing) → interpret kiya (chori ho gaya) → conclusion draw kiya (misuse ho sakta hai) → decision liya (Airtel ko call karke number block karo). Yehi loop AI agent bhi follow karta hai.

#### Reasoning kahां-kahां chahiye — Planning stage
- **Goal decomposition** — goal ko sub-steps mein todna khud reasoning ka kaam hai
- **Tool selection** — kis step ke liye kaunsa tool chahiye, ye figure out karna
- **Resource estimation** — time, dependencies, risks estimate karna

#### Reasoning kahां-kahां chahiye — Execution stage
- **Decision-making** — multiple valid options mein se choose karna (e.g., 2 candidates interview karu ya 3?)
- **Human-in-the-loop trigger** — kab khud decide karna hai, kab human se poochna hai
- **Error handling** — tool fail ho jaye (jaise LinkedIn server down) to retry karu, human ko notify karu, ya alternate platform try karu — ye decide karna

> **Gap-fill note:** "Reasoning" term aajkal LLM industry mein ek specific technical meaning bhi rakhta hai — **reasoning models** (jaise o1/o3-style ya extended thinking models) jo explicitly ek internal "chain-of-thought" generate karte hain before final answer. Video mein "reasoning" ek broader cognitive-capability ke roop mein use hua hai (jo bilkul sahi hai agentic-systems context mein), lekin ye distinguish karna helpful hai: **agentic reasoning = decision-making capability across planning+execution**, jabki **"reasoning model" = ek specific type of LLM** jo iss capability ko better perform karta hai via extended inference-time compute.

---

### 2.5 Adaptability

**Definition:** Adaptability is the agent's ability to modify its plans, strategies, and actions in response to unexpected conditions, all while staying aligned with the goal.

#### Adaptability trigger karne wale 3 reasons
1. **Tool/system failures** — jaise Calendar API down ho jaana → agent directly human se availability pooch leta hai
2. **External feedback from environment** — jaise "sirf 2 applications aaye 3 din mein" → agent JD broaden karta hai ya ads boost karta hai
3. **Mid-way goal changes** — jaise "backend engineer" ke bajaye "freelancer" hire karne ka decision

> **New concept introduced here — "Environment":** Video mein pehli baar "environment" ka concept aata hai — agent jis context mein operate karta hai (chess agent ke liye chessboard, self-driving car ke liye road+pedestrians, HR agent ke liye LinkedIn+applicants+human supervisor). Ye classical **reinforcement learning (RL)** terminology hai — agent-environment interaction loop RL se hi agentic AI mein borrow hua hai. Agar aapne kabhi RL padha ho (state, action, reward, environment), to ye concept bilkul wahi hai, bas "reward" ke bajaye yahan "goal achievement" hai.

---

### 2.6 Context Awareness

**Definition:** Context awareness is the agent's ability to understand, retain, and utilize relevant information from ongoing tasks, past interactions, user preferences, and environmental cues to make better decisions through a multi-step process.

#### Context ke types jo agent store karta hai
1. Original goal
2. Progress so far + conversation history
3. Environment state (LinkedIn pe kitne applicants aaye, ad budget kitna bacha hai)
4. Tool responses (resume parser output, calendar availability)
5. User-specific preferences (company remote candidates prefer karti hai)
6. Policies/guardrails (bina approval offer letter mat bhejo)

#### Memory ke 2 types
| Type | Kya store karta hai | Human analogy |
|---|---|---|
| **Short-term memory** | Current session ka context — messages, tool calls, immediate decisions | "Aaj mujhe 4 baje tak video shoot khatam karna hai" |
| **Long-term memory** | High-level goals, past interactions, user preferences, cross-session decisions | "Main Gurgaon mein rehta hoon" |

> **Gap-fill note — practical implementation:** Video ne bola ki details LangGraph section mein aayengi, lekin ek quick preview: LangGraph mein short-term memory typically **thread-level checkpointing** (ek conversation session ke andar state persist karna) se implement hoti hai, aur long-term memory ek **separate persistent store** (jaise vector database ya key-value store) mein implement hoti hai jo multiple sessions/threads ke across accessible hoti hai. Ye Video 1 ke RAG discussion se bhi connect hota hai — long-term memory aksar RAG-style retrieval se hi access hoti hai.

---

## 3. Five Core (High-Level) Components of an Agentic AI System

| # | Component | Role (analogy) |
|---|---|---|
| 1 | **Brain** | LLM khud — goal interpret, planning, reasoning, tool selection, natural language communication |
| 2 | **Orchestrator** | Plan ko execute karta hai — task sequencing, conditional routing, retry logic, looping, delegation *(nervous system / project manager)* |
| 3 | **Tools** | External world se interaction — API calls, DB changes, emails, RAG-based knowledge retrieval *(hands and legs)* |
| 4 | **Memory** | Short-term + long-term storage + state tracking |
| 5 | **Supervisor** | Human-in-the-loop implement karta hai — approvals, guardrail enforcement, edge-case escalation |

### 3.1 Brain (LLM)
Jobs: goal interpretation → planning (goal decomposition) → reasoning (dono stages mein) → tool selection → natural language generation/understanding.

> Note: Ye specifically **LLM-based agentic AI** ke liye hai. RL-based agents (jaise game-playing agents) mein "brain" LLM nahi hota — wo policy network hota hai. Ye playlist specifically LLM-based agentic systems (LangGraph) pe focus karegi.

### 3.2 Orchestrator
Kaam: task sequencing, conditional routing (if step-2-output == X, go to step-3, else step-4), retry logic (tool fail hua to wait-and-retry), looping/iteration, delegation (kaam human ko dena hai ya LLM ko).

> **Direct mapping to LangGraph:** Orchestrator jo yahan describe hua hai, wahi exactly **LangGraph ka core** hai — LangGraph is fundamentally a graph-based orchestration framework jahan nodes = steps, edges = conditional routing, aur graph engine = orchestrator. Isiliye playlist ka naam "Agentic AI **using LangGraph**" hai — LangGraph specifically is orchestration layer ko build karne ke liye design hua hai.

> **Side note — alternative orchestration frameworks:**
> - **CrewAI** — role-based multi-agent orchestration (jaise ek "recruiter agent," ek "screener agent" collaborate karte hain)
> - **AutoGen (Microsoft)** — conversational multi-agent orchestration, agents ek-doosre se "chat" karke tasks complete karte hain
> - **LlamaIndex Agents / Workflows** — RAG-heavy pipelines ke saath orchestration
> - **OpenAI Swarm / Assistants API** — lightweight orchestration OpenAI ecosystem ke andar
>
> LangGraph ka differentiator: **explicit state graph** with cycles — isse complex, loopy, conditional workflows (jaise HR hiring scenario) model karna easier hota hai compared to purely linear chains.

### 3.3 Tools
Kaam: external actions (API calls), DB changes, email send/receive, aur **RAG (knowledge base retrieval)** bhi ek tool hi treat hota hai is framework mein.

### 3.4 Memory
Short-term (current session — messages, tool calls, immediate decisions) + Long-term (high-level goals, past interactions, preferences, cross-session decisions) + State tracking (ab tak kitna kaam hua, kitna baaki hai).

### 3.5 Supervisor
Kaam: high-risk actions (offer letter bhejna, paid ads chalana) se pehle human ko notify + permission lena; guardrails enforce karna; edge-cases escalate karna (jaise ek non-IIT/NIT candidate ka resume bohot strong hai — guardrail se bahar jaake human ko alert karna).

> **Gap-fill note:** Video ne "Brain" ke andar bhi sub-components mention kiye (planner, evaluator) but detail mein nahi gaya — ye keh ke ki beginner-friendly video hai. Real-world production agent architectures mein ye common hai ki "Brain" khud multiple specialized LLM calls mein split ho:
> - **Planner** — candidate plans generate karta hai
> - **Evaluator/Critic** — plans ko score/evaluate karta hai
> - **Executor-reasoner** — actual step-by-step reasoning execution ke time
>
> Ye pattern kabhi-kabhi **"multi-agent within a single agent"** bhi bola jaata hai, aur LangGraph jaisi frameworks mein alag-alag nodes ke roop mein implement hota hai.

---

## 4. Quick Revision Table

| Characteristic | One-line meaning |
|---|---|
| Autonomy | Bina step-by-step instruction ke decisions leke actions le sakta hai |
| Goal-Oriented | Persistent objective ke around hi saari actions directed hoti hain |
| Planning | Goal ko structured sub-goals/steps mein todna (multiple plans → evaluate → select) |
| Reasoning | Info interpret → conclusion → decision (planning + execution dono mein) |
| Adaptability | Unexpected situations mein plan modify kar sakta hai, goal-aligned rehte hue |
| Context Awareness | Ongoing task, past interactions, preferences, environment cues yaad rakhna |

| Component | One-line meaning |
|---|---|
| Brain | LLM — interpret, plan, reason, select tools, communicate |
| Orchestrator | Plan ko sequence/route/retry/loop karke execute karwata hai |
| Tools | External world ke saath interaction (APIs, DB, email, RAG) |
| Memory | Short-term + long-term + state tracking |
| Supervisor | Human-in-the-loop, guardrails, escalations |

---

## 5. Connects to Video 1

- Video 1 mein jo "Tool-Augmented Chatbot → Agentic Chatbot" evolution dikhaya gaya tha, wahi 6 characteristics is video mein formally naam de diye gaye hain.
- Video 1 ka "RAG-based knowledge" wahi is video ke "Tools" component ke andar aata hai (RAG = ek type of tool).
- Video 1 ka "memory na hona" problem — is video ke "Context Awareness" + "Memory" component se directly maps karta hai.

Agla step: LangGraph ke saath practically ye sab components (Brain, Orchestrator, Tools, Memory, Supervisor) implement karna seekhenge.