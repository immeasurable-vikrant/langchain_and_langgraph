# Video 12: Streaming in LangGraph

**Playlist:** Agentic AI using LangGraph (CampusX, Nitish)
**Video topic:** Chatbot mein ek aur feature add karna — **Streaming** (typewriter-effect response, `.invoke()` ke bajaye `.stream()`)

> **Series recap:** Basic chatbot (Video 9) → Short-term memory (Video 9/10) → UI (Video 11) → **Streaming (is video)**

---

## 1. The Problem (Demonstrated Live)

User ne chatbot se bola: "Write a 500 word long blog on cricket."

**Kya hua:** Screen **blank** rehti hai kai seconds tak, phir **poora 500-word blog ek saath appear** ho jaata hai.

### 1.1 Two Issues with This
1. User ko **wait** karna padta hai (5-10 seconds, output ki length ke hisaab se) bina koi feedback ke
2. Jab poora text ek saath aata hai, wo **kam readable** hota hai — ChatGPT jaise tools mein hum character-by-character/token-by-token typing effect dekhte hain, jo yaha missing hai

---

## 2. What is Streaming?

> **Definition:** In LLMs, streaming means the model starts sending tokens as soon as they are generated, instead of waiting for the entire response to be ready before returning it.

**Do tareeke response return karne ke:**
| Approach | Behavior |
|---|---|
| Non-streaming | LLM poora answer sochta hai, generate karta hai, phir **ek saath** poora response deta hai |
| **Streaming** | LLM jaise-jaise answer generate karta hai, **token-by-token** turant milta rehta hai |

---

## 3. Why Streaming Matters — 5 Reasons (from video)

### 3.1 Faster Perceived Response Time
Lambe outputs (500-1000 words) generate hone mein 5-10 seconds lag sakte hain. Bina streaming ke, is duration mein screen **completely blank** rehti hai. Ek non-technical user ko lag sakta hai app **frozen/broken** hai, aur wo app band karke chala jaaye — **drop-off** ho sakta hai. Streaming turant kuch dikhata hai, isliye lagta hai system "working" hai.

### 3.2 Mimics Human-like Conversation
Streaming trust build karta hai, "alive" feel deta hai, aur user ko engaged rakhta hai — jaisa ki real ChatGPT interaction mein feel hota hai.

### 3.3 Essential for Multimodal UI (Voice Assistants)
Agar aap Alexa-jaise voice device se baat kar rahe ho aur streaming nahi hai, to device **poora response generate hone ka wait karega**, phir bolna shuru karega — jisse conversation mein ek awkward, disconnected delay feel hota hai (jaise phone call mein signal issue).

### 3.4 Better UX for Long Outputs (like Code)
Agar poora code ek saath screen pe dump ho jaaye, to samajhna mushkil hota hai ki kaha kya likha hai. Word-by-word/line-by-line streaming se code ko **follow karna easier** hota hai.

### 3.5 Cost Savings via Early Stop
Agar response pasand nahi aa raha, to user **mid-way stop** kar sakta hai. Isse baaki tokens generate hi nahi hote — jo **token-based pricing** ki wajah se seedha **cost saving** hai (LLM providers usage token-count ke basis pe charge karte hain).

### 3.6 Bonus — Streaming isn't just for final message text
Streaming sirf LLM ka final text dikhane ke liye nahi — **process updates** dikhane ke liye bhi useful hai. Example: agar AI agent movie ticket book kar raha hai, to bina streaming ke 1 minute tak kuch nahi dikhega, jisse user uncertain/anxious feel karega. Streaming se aap step-by-step updates de sakte ho: "Opening BookMyShow" → "Selecting movie" → "Selecting seats" → "Selecting payment mode" → "Processing payment".

> **Gap-fill/forward-reference note:** Ye last point (process updates streaming) is video mein implement nahi hua — instructor ne explicitly bola ki ye tab aayega jab **Agentic applications with Tools** banayenge. Uske liye LangGraph ke **different stream modes** (`updates`, `values`, `custom`) relevant honge — is video mein sirf `messages` mode use hua.

---

## 4. Core Code Change: `.invoke()` → `.stream()`

### 4.1 What `.stream()` Returns — A Generator
```python
result = chatbot.stream(...)
print(type(result))  # <class 'generator'>
```

> **Python Generator recap (as explained in video):** In Python, a generator is a special type of iterator that allows you to generate values on the fly, one at a time, using the `yield` keyword instead of `return`. Ek generator ke output ko access karne ke liye aap uske upar **loop** chalate ho.

### 4.2 Basic Backend Usage
```python
from langchain_core.messages import HumanMessage

CONFIG = {"configurable": {"thread_id": "thread-1"}}

for message_chunk, metadata in chatbot.stream(
    {"messages": [HumanMessage(content="What is the recipe to make pasta")]},
    config=CONFIG,
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end=" ")
```

### 4.3 Understanding the Return Structure — `(message_chunk, metadata)`
`.stream()` se har iteration mein **do cheezein** milti hain:
1. **`message_chunk`** — jahan actual message content chunk store hai (`.content` attribute)
2. **`metadata`** — additional info about that chunk

> **Gap-fill note:** Video mein `metadata` ka exact content detail mein nahi explain hua. Practically, `metadata` mein typically ye hota hai: kaunse node se ye chunk aaya (`langgraph_node`), kya checkpoint/thread info hai, aur kabhi-kabhi model-specific info (jaise token usage). Is video mein sirf `message_chunk.content` use hua, `metadata` ignore kiya gaya — lekin agentic apps mein jaha multiple nodes/tools involved hote hain, `metadata` se ye pata chalta hai ki **kaunsa node currently stream kar raha hai**, jo debugging/UI-labeling ke liye useful hota hai.

### 4.4 The `stream_mode` Parameter
LangGraph mein multiple streaming modes available hain:

| Mode | Kab use hota hai |
|---|---|
| `messages` | LLM ka token-by-token response stream karne ke liye *(is video ka focus)* |
| `updates` | Har node ke baad state mein hue **updates** dikhane ke liye |
| `values` | Har step ke baad **poori state ki current value** dikhane ke liye |
| `custom` | Developer-defined custom streaming events ke liye |

> **Gap-fill note — jab kaunsa mode use hoga:** Video ne bola ki `updates`/`values`/`custom` future videos mein (Tools/Agentic apps banate waqt) explore honge. Quick preview of typical use:
> - `messages` — chat UI mein LLM ka text stream karna (is video ka use case)
> - `updates` — agent ke "step updates" dikhane ke liye jaisa Section 3.6 mein describe hua (jaise "searching...", "found result...")
> - `values` — poore workflow ka progress track karna (debugging ke liye achha, kyunki har step ka poora snapshot milta hai)
> - `custom` — jab aap khud specific events emit karna chahte ho jo standard modes cover nahi karte (jaise custom progress bars)

---

## 5. Frontend Integration (Streamlit)

### 5.1 New Streamlit Component: `st.write_stream`
Streamlit ke docs mein multiple chat-related elements hain:
- `st.chat_input` (already used — Video 11)
- `st.chat_message` (already used — Video 11)
- `st.status` — agent-type apps ke liye status updates dikhane ke liye (mentioned, not used yet)
- **`st.write_stream`** *(is video ka focus)* — "Write generators and streams to the app with a typewriter effect"

> `st.write_stream` ek **generator** accept karta hai aur uska poora UI handling (typewriter effect render karna) khud kar leta hai — developer ko manually loop chalake print karne ki zaroorat nahi.

### 5.2 Code Change in Frontend

**Before (Video 11's non-streaming approach):**
```python
response = chatbot.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config=CONFIG
)
ai_message = response["messages"][-1].content

st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
with st.chat_message("assistant"):
    st.text(ai_message)
```

**After (streaming version):**
```python
with st.chat_message("assistant"):
    ai_message = st.write_stream(
        message_chunk.content for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages"
        )
    )

st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
```

### 5.3 Key Points
- `st.write_stream()` ko ek **generator expression** diya gaya hai (`message_chunk.content for ... in ...`) — ye directly `chatbot.stream()` ke output ko consume karta hai
- `st.write_stream()` **return value** — poora accumulated final text return karta hai (jo `ai_message` mein store hota hai) — isse aap baad mein `session_state` mein history ke liye save kar sakte ho
- **Minimal changes needed:** Sirf assistant-response-generation wala hissa change hua; baaki sab (user input handling, history loading, session_state) Video 11 se **as-is** reused hua

### 5.4 Bug Encountered in Video (worth noting)
Instructor ne temporarily **hardcoded** message ("What is the recipe to make pasta") test karne ke liye backend mein, lekin fir frontend integrate karte waqt wahi hardcoded string bhool ke reh gayi thi (`user_input` ki jagah). Isse alag-alag prompts type karne pe bhi wahi purana output aata raha ("cricket in India" jaisa unrelated output). **Lesson:** Backend test karte waqt use hue temporary/hardcoded values ko integration se pehle **wapas dynamic variable se replace** karna na bhoolen — ek common aur easy-to-miss debugging trap.

---

## 6. Gaps Filled / Additional Context

### 6.1 Generators — quick refresher (referenced but not fully explained in video)
```python
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()
for value in gen:
    print(value)  # 1, 2, 3 — one at a time, computed on demand
```
- `yield` vs `return`: `return` ek function ko turant khatam karke ek value deta hai; `yield` function ko **pause** kar deta hai aur value deta hai, phir jab dobara call ho to wahi se resume hota hai
- Ye **memory-efficient** hota hai — poori list ek saath memory mein nahi rakhni padti, values on-demand generate hoti hain
- LangGraph ka `.stream()` internally isi pattern ko use karta hai — LLM se aane wale tokens ko real-time yield karta hai, poore response ka wait nahiं karta

### 6.2 Streaming vs Server-Sent Events (SSE) / WebSockets — broader context
Video ne is technical layer ko touch nahi kiya, lekin conceptually samajhna useful hai: jab aap production mein (Streamlit ke bina, jaise React frontend + FastAPI backend) streaming implement karte ho, to backend se frontend tak tokens bhejne ke liye typically **Server-Sent Events (SSE)** ya **WebSockets** use hote hain. Streamlit is complexity ko abstract kar deta hai (`st.write_stream` internally ye sab handle karta hai), isliye demo/prototype ke liye itna simple lagta hai — production-grade custom UI banate waqt ye underlying networking layer explicitly implement karni padegi.

### 6.3 Cost-saving via early stop — practical implication
Section 3.5 ka point (mid-response stop karke tokens/cost bachana) sirf UX feature nahi hai — production LLM applications mein ye ek real **cost optimization strategy** hai. Agar aap khud ka LLM-based product bana rahe ho, "stop generating" button dena sirf UX ke liye nahi, balki **billing** ke liye bhi important hai, especially agar aap per-token pricing wale provider (OpenAI, Anthropic, etc.) use kar rahe ho.

### 6.4 Connecting to earlier videos
- **Video 10's `thread_id`/checkpointer requirement** — yahan bhi `config=CONFIG` (with `thread_id`) `.stream()` call mein zaroori hai, exactly `.invoke()` jaisa hi.
- **Video 11's `st.session_state`** — is video mein bhi history persistence ke liye same `session_state` pattern reuse hua; sirf ai_message generate karne ka tareeka (`write_stream` ke through) badla.

---

## 7. Quick Revision Table

| Concept | One-liner |
|---|---|
| Streaming | Response ko token-by-token turant bhejna, poora complete hone ka wait kiye bina |
| `.invoke()` → `.stream()` | Non-streaming se streaming mein switch karne ka core backend change |
| Return type of `.stream()` | Python **generator** |
| `(message_chunk, metadata)` | Har iteration mein milne wale 2 objects — content aur additional info |
| `stream_mode="messages"` | Token-by-token LLM text ke liye correct mode |
| Other stream_modes | `updates`, `values`, `custom` — future agentic-app videos mein cover honge |
| `st.write_stream()` | Streamlit component — generator leke automatically typewriter-effect UI render karta hai |
| 5 key benefits | Faster perceived response, human-like feel, essential for voice UI, better long-output readability, cost savings via early stop |