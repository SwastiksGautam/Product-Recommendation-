# app.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json, re, os, ast
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Import recommendation logic
from llm_agent import get_recommendations

# -------------------- Setup --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Unified SHL MCP + Recommendation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- Load Product Data --------------------
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "shl_all_products_details.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)
df = df.fillna("")

# Convert embedding column if it exists
if "embedding" in df.columns:
    df["embedding"] = df["embedding"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# -------------------- Product Endpoints (MCP) --------------------
class Product(BaseModel):
    name: str
    link: str
    Description: str
    Job_levels: str
    Languages: str
    Assessment_length: str
    Test_Type: str
    Remote_Testing: str

@app.get("/products", response_model=list[Product])
def get_all_products():
    """Return all SHL products."""
    return df.to_dict(orient="records")


@app.get("/filter", response_model=list[Product])
def filter_products(
    job_level: Optional[str] = None,
    test_type: Optional[str] = None,
    remote: Optional[str] = None,
    language: Optional[str] = None,
    keyword: Optional[str] = None
):
    """Filter products based on parameters."""
    filtered = df.copy()
    if job_level:
        filtered = filtered[filtered["Job levels"].str.contains(job_level, case=False, na=False)]
    if test_type:
        filtered = filtered[filtered["Test Type"].str.contains(test_type, case=False, na=False)]
    if remote:
        filtered = filtered[filtered["Remote Testing"].str.lower().str.contains(remote.lower(), na=False)]
    if language:
        filtered = filtered[filtered["Languages"].str.contains(language, case=False, na=False)]
    if keyword:
        filtered = filtered[filtered["Description"].str.contains(keyword, case=False, na=False)]
    return filtered.to_dict(orient="records")


@app.get("/sort", response_model=list[Product])
def sort_products(by: str = Query("name"), ascending: bool = True):
    """Sort products by a specific field."""
    valid_fields = {
        "name": "name",
        "job_level": "Job levels",
        "assessment_length": "Assessment length"
    }
    field = valid_fields.get(by.lower())
    if not field:
        raise HTTPException(status_code=400, detail="Invalid sort field")
    return df.sort_values(by=field, ascending=ascending).to_dict(orient="records")


@app.get("/semantic_search", response_model=list[Product])
def semantic_search(query: str, top_n: int = 5):
    """Perform semantic search based on embeddings."""
    keywords = [q.strip() for q in query.split(",") if q.strip()]
    if not keywords:
        return []

    def cosine_sim(a, b): 
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    aggregated_scores = np.zeros(len(df))
    for kw in keywords:
        q_emb = client.embeddings.create(
            model="text-embedding-ada-002", 
            input=kw
        ).data[0].embedding
        df["similarity"] = df["embedding"].apply(
            lambda x: cosine_sim(np.array(x), np.array(q_emb))
        )
        aggregated_scores += df["similarity"].values

    df["aggregated_similarity"] = aggregated_scores / len(keywords)
    top_products = df.sort_values(by="aggregated_similarity", ascending=False).head(top_n)
    return top_products.to_dict(orient="records")

# -------------------- Recommendation Endpoint --------------------
class RecommendedAssessment(BaseModel):
    url: str
    name: str
    adaptive_support: Optional[str] = ""
    description: Optional[str] = ""
    duration: Optional[str] = ""
    remote_support: Optional[str] = ""
    test_type: Optional[str] = ""

class RecommendationResponse(BaseModel):
    recommended_assessments: List[RecommendedAssessment]

@app.get("/recommend", response_model=RecommendationResponse)
def recommend_assessments(query: str):
    """Main recommendation endpoint that calls LLM + MCP logic."""
    recommendations = get_recommendations(query)
    if not recommendations:
        raise HTTPException(status_code=404, detail="No recommendations found")

    if len(recommendations) < 5:
        recommendations *= 2
    recommendations = recommendations[:10]

    return {"recommended_assessments": recommendations}

# -------------------- Run Server --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
