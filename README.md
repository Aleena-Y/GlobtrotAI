# 🌍 GlobtrotAI

**AI-Powered Country Recommender for Global Opportunities**

GlobtrotAI is an intelligent AI system that recommends **the best countries to pursue your goals** — whether in **education, research, or innovation** — using a **Retrieval-Augmented Generation (RAG)** pipeline powered by **Ollama (Gemma 3:1B)** and **FAISS-based semantic search**.

---

## 🧭 Overview

GlobtrotAI connects a **FastAPI backend** (for inference and data retrieval) with a **React frontend** (for user interaction).
The system uses **Gemma 3:1B** locally via Ollama to generate context-aware, explainable country recommendations.

---

## 🚀 Features

* 🔍 **RAG with FAISS** – Retrieves relevant country data for precise recommendations.
* 🧠 **Local LLM (Gemma 3:1B)** – Fast, lightweight reasoning via Ollama.
* 🧩 **Explainable Output** – Every recommendation comes with a short rationale.
* 🧭 **Interactive Web UI** – Simple chat interface built in React.
* ⚙️ **Modular Architecture** – Easy to plug in new datasets or domains.

---

## 🏗️ Project Structure

```
GlobtrotAI/
├── backend/
│   ├── app.py                  # FastAPI backend
│   ├── rag_pipeline.py         # Retrieval-Augmented Generation logic
│   ├── generate_embeddings.py  # Embedding generation script
│   ├── query_ollama.py         # Handles requests to Ollama API
│   ├── requirements.txt
│   └── data/
│       └── countries.json
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── ChatBox.jsx
│   │   │   └── CountryCard.jsx
│   │   └── api/
│   │       └── client.js
│   ├── package.json
│   └── vite.config.js
│
├── embeddings/                 # FAISS index files
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/GlobtrotAI.git
cd GlobtrotAI
```

---

### 2. Backend Setup

#### Create Virtual Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

#### Install Requirements

```bash
pip install -r requirements.txt
```

#### Example `requirements.txt`

```
fastapi
uvicorn
faiss-cpu
sentence-transformers
requests
pydantic
```

---

### 3. Install and Configure Ollama

#### Install Ollama

Download from [https://ollama.ai](https://ollama.ai) and install.

#### Pull the Gemma Model

```bash
ollama pull gemma:1b
```

#### Run Ollama in Background

```bash
ollama serve
```

Confirm it’s working:

```bash
curl http://localhost:11434/api/generate -d '{"model":"gemma:1b","prompt":"Hello"}'
```

---

### 4. Generate Embeddings

Before first use, embed your dataset:

```bash
python generate_embeddings.py
```

This creates a FAISS index in `/embeddings`.

---

### 5. Run FastAPI Backend

```bash
uvicorn app:app --reload
```

API will run on `http://localhost:8000`

Example endpoint:

```
POST /recommend
{
  "query": "Which country is best for space research?"
}
```

---

### 6. Frontend Setup (React)

```bash
cd ../frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173` (Vite default).

Make sure the API URL in `client.js` points to `http://localhost:8000`.

---

## 🧠 Example Prompt

```
User: Suggest countries suitable for renewable energy startups.
```

**GlobtrotAI Output:**

> “Germany and Denmark lead in clean energy innovation.
> Germany’s research funding is high, while Denmark supports small-scale energy ventures.”

---

## 🔄 API Endpoints

| Method | Endpoint     | Description                             |
| ------ | ------------ | --------------------------------------- |
| POST   | `/recommend` | Generates country recommendations       |
| GET    | `/health`    | Verifies server and Ollama connectivity |
| POST   | `/embed`     | Regenerates FAISS embeddings            |

---

## 🧰 Troubleshooting

**1. Ollama not returning any output?**
Run:

```bash
ollama serve
```

and test:

```bash
curl http://localhost:11434/api/generate -d '{"model":"gemma:1b","prompt":"test"}'
```

**2. Backend gives empty results?**
Ensure FAISS index exists and embeddings are generated correctly.

**3. Frontend can’t reach backend?**
Open `/frontend/src/api/client.js` and confirm:

```js
const BASE_URL = "http://localhost:8000";
```

---

## 🛣️ Roadmap

* [ ] Add country scoring visualization dashboard
* [ ] Integrate real-time global datasets (research output, cost of living, etc.)
* [ ] Deploy on Docker and connect to cloud Ollama servers
* [ ] Add authentication for user profiles

---

## 🧾 License

Licensed under the **MIT License** — free to modify, use, and build upon.

---

## ✨ Author

**Aleena Yogindar**
Engineering Student @ VIT Chennai
Building systems that make AI think globally and recommend intelligently.
