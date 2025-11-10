# app.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os, ast, json, re, uvicorn

# Import recommendation logic from llm_agent
from llm_agent import get_recommendations

# -------------------- Setup --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Missing OPENAI_API_KEY in environment variables!")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Unified SHL MCP + Recommendation Backend")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- Health Check --------------------
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def root():
    return {"message": "🚀 Product Recommendation API is live!"}

# -------------------- Load Product Data --------------------
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "shl_all_products_details.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)
string_cols = ['name', 'link', 'Description', 'Job_levels', 'Languages',
               'Assessment_length', 'Test_Type', 'Remote_Testing', 'embedding']

for col in string_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Strip whitespace
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Rename columns for consistency
df.rename(columns={
    "Job levels": "Job_levels",
    "Assessment length": "Assessment_length",
    "Test Type": "Test_Type",
    "Remote Testing": "Remote_Testing"
}, inplace=True)

# Convert or compute embeddings if missing
if 'embedding' not in df.columns:
    print("No embeddings found. Computing embeddings now...")
    embeddings = []
    for desc in df['Description']:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=str(desc))
        embeddings.append(resp.data[0].embedding)
    df['embedding'] = embeddings
    df.to_csv(csv_path, index=False)
else:
    df['embedding'] = df['embedding'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

# -------------------- Pydantic Models --------------------
class Product(BaseModel):
    name: str
    link: str
    Description: str
    Job_levels: str
    Languages: str
    Assessment_length: str
    Test_Type: str
    Remote_Testing: str

class AssessmentResource(BaseModel):
    url: str                       # Valid URL to the assessment resource
    name: str                      # Name of the assessment
    adaptive_support: str           # "Yes" or "No"
    description: str                # Detailed description of the assessment
    duration: Optional[int]         # Duration of the assessment in minutes
    remote_support: str             # "Yes" or "No"
    test_type: List[str]            # Categories or types of the assessment

class RecommendationResponse(BaseModel):
    recommended_assessments: List[AssessmentResource]


# -------------------- Endpoints --------------------
@app.get("/products", response_model=List[Product])
def get_all_products():
    """Return all SHL products."""
    return df.to_dict(orient="records")

@app.get("/filter", response_model=List[Product])
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
        filtered = filtered[filtered["Job_levels"].str.contains(job_level, case=False, na=False)]
    if test_type:
        filtered = filtered[filtered["Test_Type"].str.contains(test_type, case=False, na=False)]
    if remote:
        filtered = filtered[filtered["Remote_Testing"].str.lower().str.contains(remote.lower(), na=False)]
    if language:
        filtered = filtered[filtered["Languages"].str.contains(language, case=False, na=False)]
    if keyword:
        filtered = filtered[filtered["Description"].str.contains(keyword, case=False, na=False)]
    return filtered.to_dict(orient="records")

@app.get("/sort", response_model=List[Product])
def sort_products(by: str = Query("name"), ascending: bool = True):
    """Sort products by a specific field."""
    valid_fields = {
        "name": "name",
        "job_level": "Job_levels",
        "assessment_length": "Assessment_length"
    }
    field = valid_fields.get(by.lower())
    if not field:
        raise HTTPException(status_code=400, detail="Invalid sort field")
    return df.sort_values(by=field, ascending=ascending).to_dict(orient="records")

@app.get("/semantic_search", response_model=List[Product])
def semantic_search(query: str, top_n: int = 5):
    """Perform semantic search based on embeddings."""
    try:
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

        return top_products.drop(columns=["similarity", "aggregated_similarity"]).to_dict(orient="records")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(body: dict):
    """
    Main recommendation endpoint (POST) that accepts JSON body:
    {
        "query": "job description text",
        "input_type": "text"  # or "url"
    }
    """
    query = body.get("query")
    input_type = body.get("input_type", "text")

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in request body")
    if input_type not in ["text", "url"]:
        raise HTTPException(status_code=400, detail="input_type must be 'text' or 'url'")

    raw_recs = get_recommendations(query, input_type=input_type)
    if not raw_recs:
        raise HTTPException(status_code=404, detail="No recommendations found")

    # Deduplicate
    seen = set()
    unique_recs = []
    for rec in raw_recs:
        name = rec.get("name", "").strip().lower()
        url = rec.get("url", "").strip().lower()
        key = f"{name}|{url}"
        if key not in seen and name and url:
            unique_recs.append(rec)
            seen.add(key)

    # Ensure at least 5 and max 10
    if len(unique_recs) < 5:
        for rec in raw_recs:
            name = rec.get("name", "").strip().lower()
            url = rec.get("url", "").strip().lower()
            key = f"{name}|{url}"
            if key not in seen and name and url:
                unique_recs.append(rec)
                seen.add(key)
            if len(unique_recs) >= 5:
                break

    # Transform each recommendation to strict schema
    def transform_rec(r):
        # Duration as integer (optional)
        dur = r.get("duration")
        if dur is not None:
            try:
                dur = int(dur)
            except:
                dur = None

        # Map test_type codes to descriptive list
        test_type_map = {
            "A": "Ability & Aptitude",
            "B": "Biodata & Situational Judgement",
            "C": "Competencies",
            "D": "Development & 360",
            "E": "Assessment Exercises",
            "K": "Knowledge & Skills",
            "P": "Personality & Behaviour",
            "S": "Simulations"
        }

        tt = r.get("test_type", [])
        if isinstance(tt, str):
            tt = [t.strip().upper() for t in tt.split(",") if t.strip()]
        if isinstance(tt, list):
            tt = [test_type_map.get(t, t) if isinstance(t, str) else t for t in tt]
        else:
            tt = []
        
        tt_str = ", ".join(tt)

        return {
            "url": r.get("url", ""),
            "name": r.get("name", ""),
            "adaptive_support": "Yes" if str(r.get("adaptive_support","")).lower() == "yes" else "No",
            "description": r.get("description", ""),
            "remote_support": "Yes" if str(r.get("remote_support","")).lower() == "yes" else "No",
            "duration": dur,
            "test_type": tt  # This is now a list of strings
        }



    final_recs = [transform_rec(r) for r in unique_recs[:10]]

    return {"recommended_assessments": final_recs}


# -------------------- Run Server --------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=10000, reload=True)
