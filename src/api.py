# api.py - FastAPI wrapper for the recommender using Ollama (gemma:3b)
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os
import logging

from rag_country_recommender_hybrid import (
    load_dataset, build_country_text, embed_texts_local,
    build_index, recommend_countries
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS (allow all origins during local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load dataset & embeddings ----
logger.info("Loading dataset...")
df = load_dataset()
docs = [build_country_text(row) for _, row in df.iterrows()]

emb_cache_path = "embeddings.npy"

if os.path.exists(emb_cache_path):
    logger.info("Loading cached embeddings...")
    emb_matrix = np.load(emb_cache_path)
else:
    logger.info("Computing embeddings and caching to embeddings.npy ...")
    emb_matrix = embed_texts_local(docs)
    np.save(emb_cache_path, emb_matrix.astype("float32"))

logger.info("Building FAISS index...")
index = build_index(emb_matrix)

# ---- API endpoint ----
@app.get("/recommend")
def recommend(query: str = Query(..., min_length=1), model: str = "gemma3:1b"):
    """
    query: user query string
    model: optional Ollama model name (default gemma:3b)
    """
    try:
        # pass through model by updating call_generation's default usage:
        result = recommend_countries(query, index, docs, emb_matrix, df)

        original_rationale = result.get("rationale", {}) or {}
        original_tips = result.get("tips", {}) or {}

        # produce normalized lowercase-key dicts for frontend convenience
        rationale_dict = {}
        tips_dict = {}

        for k, v in original_rationale.items():
            if k is None:
                continue
            rationale_dict[str(k).strip().lower()] = v

        for k, v in original_tips.items():
            if k is None:
                continue
            tips_dict[str(k).strip().lower()] = v

        # also ensure entries for recommended country spellings (original-case)
        recommendations = result.get("recommendations", []) or []
        normalized_recommendations = []
        for rc in recommendations:
            normalized_recommendations.append(rc)

        response = {
            "recommendations": normalized_recommendations,
            "rationale": rationale_dict,
            "tips": tips_dict,
            "llm_output": result.get("llm_output", ""),
            "parse_error": result.get("parse_error", None),
            "explanation": result.get("explanation", None),
            "retrieved": result.get("retrieved", []),
        }

        logger.info(f"Query: {query}")
        logger.info(f"Recommendations: {response['recommendations']}")
        return response

    except Exception as e:
        logger.exception("Recommendation failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
