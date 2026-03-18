# 📈 Stock Assistant (AI-Powered)

An intelligent AI Stock Market Assistant built using modern LLMs and agentic workflows. The application enables users to interact with financial data conversationally, combining real-time insights, research, and knowledge retrieval in a single interface.

---

# 🚀 Purpose

The goal of this project is to create a **smart financial assistant** that can:

* Help users analyze stocks in real-time
* Provide latest market insights
* Assist with financial learning
* Combine multiple data sources into a single conversational interface

---

# 🧠 Tech Stack

### 🔹 Core AI & Agent

* LangChain (ReAct Agent framework)
* Google Gemini (`gemini-2.5-flash-lite`)

### 🔹 Data Sources

* Yahoo Finance (via `yfinance`)
* Apify (Yahoo Finance scraper)
* Google News RSS
* ArXiv API
* Wikipedia (for financial knowledge)

### 🔹 Vector Database & NLP

* FAISS (vector store)
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* BeautifulSoup (HTML parsing)

### 🔹 Frontend

* Streamlit (chat-based UI)

### 🔹 Other Tools

* dotenv (environment variables)
* feedparser (RSS parsing)

---

# ⚙️ Workflow

The system follows a **tool-based intelligent workflow using a ReAct Agent**:

### 1. User Input

User asks a question via Streamlit chat interface.

---

### 2. Agent Reasoning

The LangChain ReAct agent:

* Understands user intent
* Decides which tool(s) to use

---

### 3. Tool Execution

* **Ticker Resolution:** Converts company names into ticker symbols using `yfinance`
* **Real-Time Data Retrieval:** Fetches live stock data using Apify Yahoo Finance scraper
* **News Aggregation:** Collects recent stock-related news via Google News RSS
* **Research Retrieval:** Queries ArXiv for academic finance papers
* **Knowledge Search:** Performs semantic search over financial concepts using FAISS + Wikipedia

---

### 4. Response Generation

* Agent gathers tool outputs
* Gemini LLM generates a final, clear response

---

### 5. UI Display

* Streamlit renders chat interface
* Maintains conversation history

---

# 📂 Project Structure

```id="053k3j"
├── agent.py              # Agent + LLM setup
├── my_tools.py           # All custom tools
├── main.py               # Streamlit app
├── requirements.txt      # Dependencies
├── .env                  # API keys
└── stock_faiss_db_0/     # Vector DB (auto-created)
```

---

# 🔑 Environment Variables

Create a `.env` file:

```id="pgoyi8"
gemini_api_key=YOUR_GEMINI_API_KEY
APIFY_API_TOKEN=YOUR_APIFY_API_TOKEN
```

---

# 💻 How to Run Locally

### 1️⃣ Install Dependencies

```id="5dpix9"
pip install -r requirement.txt
```

---

### 2️⃣ Run the App

#### ▶️ Normal Run

```id="xmb2eg"
streamlit run main.py
```

#### 🧾 Run with Logs (Recommended for Debugging)

```id="p4pkjb"
python -u -m streamlit run main.py | Tee-Object -FilePath my_agent_logs.txt
```

---

# ⚠️ Notes

* First run of `stock_website_search` will create a FAISS database
* Requires active internet connection
* API rate limits may apply (Gemini / Apify)

---
