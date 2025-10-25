"""
rag_country_recommender_hybrid.py

RAG + LIME explainability using:
- Local sentence-transformers embeddings
- Ollama (gemma:3b) via HTTP API
- FAISS retrieval
- LIME surrogate explanations
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lime import lime_tabular
import logging
import requests
import re
import json
# Ollama API endpoint (default)
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- CONFIG ----------
CSV_PATH = "countries_dataset.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 6
EMB_CACHE = "embeddings.npy"
LIME_HTML_OUT = "lime_explanation.html"

# Load embedding model
try:
    emb_model = SentenceTransformer(EMBED_MODEL)
except Exception as e:
    logger.error(f"Failed to load embedding model '{EMBED_MODEL}': {e}")
    raise

NUMERIC_FEATURES = [
    "Cost of Living Index",
    "Rent Index",
    "Cost of Living Plus Rent Index",
    "Groceries Index",
    "Restaurant Price Index",
    "Local Purchasing Power Index",
    "median_salary",
    "average_salary",
    "lowest_salary",
    "highest_salary",
]

# -------- dataset --------
def load_dataset(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    expected_cols = ["Country"] + NUMERIC_FEATURES + ["continent_name"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df["Country"] = df["Country"].astype(str).str.strip()
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df

def build_country_text(row):
    parts = [
        f"{row['Country']} ({row['continent_name']})",
        f"Cost of Living: {row['Cost of Living Index']}",
        f"Rent Index: {row['Rent Index']}",
        f"Cost of Living + Rent: {row['Cost of Living Plus Rent Index']}",
        f"Groceries: {row['Groceries Index']}",
        f"Restaurants: {row['Restaurant Price Index']}",
        f"Local Purchasing Power: {row['Local Purchasing Power Index']}",
        f"Salaries — median: {row['median_salary']}, avg: {row['average_salary']}, range: {row['lowest_salary']}-{row['highest_salary']}",
    ]
    return ". ".join(parts)

# -------- embeddings & retrieval --------
def embed_texts_local(texts):
    embs = emb_model.encode(texts, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")
    return embs

def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index

def retrieve(query, index, docs, k=TOP_K):
    q_emb = embed_texts_local([query])[0]
    D, I = index.search(np.array([q_emb]).astype("float32"), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if int(idx) < 0 or int(idx) >= len(docs):
            continue
        results.append({"doc_idx": int(idx), "score": float(score), "text": docs[int(idx)]})
    return results, q_emb

# -------- Prompt builder --------
def build_prompt(query, retrieved):
    blocks = []
    for i, r in enumerate(retrieved):
        blocks.append(f"Candidate {i+1}: {r['text']}")
    context = "\n\n".join(blocks)

    prompt = (
        "You are an assistant that recommends countries based on a user's requirements.\n"
        f"User query: '''{query}'''\n\n"
        "Context (retrieved country summaries):\n"
        + context
        + "\n\n"
        "Task: Recommend the best 1-6 countries that match the user's request.\n"
        "For each recommended country, provide:\n"
        " - A one-sentence rationale citing one or two numeric fields (e.g. Cost of Living, Rent Index, salaries).\n"
        " - A one-line practical budget tip.\n\n"
        "IMPORTANT: Output MUST be valid JSON only (no extra text). The JSON object must have exactly these keys:\n"
        " - recommendations: list of country names\n"
        " - rationale: object mapping country -> rationale string\n"
        " - tips: object mapping country -> tip string\n\n"
        "Example:\n"
        '{"recommendations": ["India", "Vietnam"], "rationale": {"India":"Low cost of living (24.5).","Vietnam":"Affordable rent index."}, "tips": {"India":"Trains are affordable.","Vietnam":"Street food is cheap."}}\n'
        "Only return valid JSON."
    )
    return prompt

# -------- JSON extractor --------


def _extract_json_from_text(text: str):
    if not isinstance(text, str):
        return None
    
    # 1. Remove Markdown code fences like ```json ... ```
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text, flags=re.IGNORECASE).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    
    # 2. Extract the first JSON object
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    candidate = None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                break
    
    if not candidate:
        return None
    
    # 3. Try strict parse first
    try:
        return json.loads(candidate)
    except Exception:
        pass
    
    # 4. Fallback: attempt to repair common issues
    #    - remove trailing commas
    cleaned = re.sub(r",\s*}", "}", candidate)
    cleaned = re.sub(r",\s*]", "]", cleaned)
    
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# -------- Call Ollama over HTTP --------
def call_generation(prompt, model_name="gemma3:1b", max_new_tokens=400):
    """
    Send a prompt to Ollama and return the model's response text.
    """
    payload = {
        "model": model_name,  # gemma:3b must be pulled via `ollama pull gemma:3b`
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_new_tokens
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        # Ollama returns a structure that typically has "response" or similar
        # We'll try common keys
        if isinstance(data, dict):
            # If the API returns a "response" string
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
            # If it returns a list of chunks or choices
            if "choices" in data:
                try:
                    # join choice texts if present
                    return " ".join([c.get("message", {}).get("content", "") for c in data["choices"] if isinstance(c, dict)])
                except Exception:
                    pass
            # fallback to raw text of dict
            return json.dumps(data)
        return str(data)
    except Exception as e:
        return f"__OLLAMA_GEN_ERROR__ {e}"

# -------- explainability (LIME) --------
def fit_surrogate(df_meta, retrieved, embeddings, q_emb):
    rows, ys = [], []
    for r in retrieved:
        idx = r["doc_idx"]
        row = df_meta.iloc[idx]
        rows.append([row[f] for f in NUMERIC_FEATURES])
        y = float(np.dot(q_emb, embeddings[idx]))
        ys.append(y)
    X, y = np.array(rows, dtype=float), np.array(ys, dtype=float)
    if X.shape[0] < 12:
        rng = np.random.RandomState(0)
        extra, extra_y = [], []
        for _ in range(200):
            j = rng.randint(0, max(1, X.shape[0]))
            noise = rng.normal(scale=0.03, size=X.shape[1])
            x_new = X[j] + noise * np.maximum(np.abs(X[j]), 1.0)
            y_new = y[j] + rng.normal(scale=0.005)
            extra.append(x_new)
            extra_y.append(y_new)
        X = np.vstack([X, np.array(extra)])
        y = np.concatenate([y, np.array(extra_y)])
    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    return scaler, model, X

def explain_with_lime(scaler, model, instance_raw, training_data_raw):
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=training_data_raw.astype(float),
        feature_names=NUMERIC_FEATURES,
        mode="regression",
    )
    def predict_fn(x_raw):
        return model.predict(scaler.transform(x_raw))
    exp = explainer.explain_instance(instance_raw.astype(float), predict_fn, num_features=6)
    contributions = exp.as_list()
    pred_val = float(predict_fn(instance_raw.reshape(1, -1))[0])
    return contributions, pred_val, exp

# -------- main pipeline --------
def recommend_countries(query, index, docs, embeddings, df_meta, do_explain=True):
    retrieved, q_emb = retrieve(query, index, docs, k=TOP_K)
    prompt = build_prompt(query, retrieved)
    llm_output = call_generation(prompt)

    recommendations = []
    rationale = {}
    tips = {}
    parse_error = None

    if isinstance(llm_output, str) and llm_output.startswith("__OLLAMA_GEN_ERROR__"):
        parse_error = llm_output
    else:
        parsed = None
        try:
            if isinstance(llm_output, str):
                parsed = _extract_json_from_text(llm_output)
            elif isinstance(llm_output, dict):
                parsed = llm_output
            if parsed is None:
                raise ValueError("Generation did not return valid JSON")
            recs = parsed.get("recommendations", [])
            if not isinstance(recs, list):
                raise ValueError("recommendations is not a list")
            cleaned = []
            for r in recs:
                if not isinstance(r, str):
                    continue
                name = r.strip()
                matched = df_meta[df_meta["Country"].str.lower() == name.lower()]
                if not matched.empty:
                    cleaned.append(matched.iloc[0]["Country"])
                else:
                    matched = df_meta[df_meta["Country"].str.lower().str.contains(name.lower())]
                    if not matched.empty:
                        cleaned.append(matched.iloc[0]["Country"])
                    else:
                        cleaned.append(name)
            recommendations = cleaned
            rationale = parsed.get("rationale", {}) or {}
            tips = parsed.get("tips", {}) or {}
        except Exception as e:
            parse_error = f"Failed to parse generation JSON: {e}"

    if not recommendations:
        recommendations = []
        for r in retrieved:
            idx = r["doc_idx"]
            recommendations.append(df_meta.iloc[idx]["Country"])

    explanation = None
    if do_explain and retrieved:
        try:
            scaler, surrogate_model, X_train_raw = fit_surrogate(df_meta, retrieved, embeddings, q_emb)
            top_idx = retrieved[0]["doc_idx"]
            instance_raw = np.array([df_meta.iloc[top_idx][f] for f in NUMERIC_FEATURES], dtype=float)
            contributions, pred_val, lime_exp = explain_with_lime(scaler, surrogate_model, instance_raw, X_train_raw)
            html = lime_exp.as_html()
            try:
                with open(LIME_HTML_OUT, "w", encoding="utf-8") as f:
                    f.write(html)
            except Exception:
                pass
            explanation = {
                "top_country": df_meta.iloc[top_idx]["Country"],
                "pred_similarity_estimate": float(pred_val),
                "lime_contributions": contributions,
                "lime_html": html,
            }
        except Exception as e:
            explanation = {"error": f"LIME failed: {e}"}

    result = {
        "recommendations": recommendations,
        "rationale": rationale,
        "tips": tips,
        "llm_output": llm_output,
        "retrieved": retrieved,
        "explanation": explanation,
        "parse_error": parse_error,
    }
    return result
