from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from openai import OpenAI
import ast

from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Get the key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------- Load CSV and Compute Embeddings -----------------
BASE_DIR = os.path.dirname(__file__)  # backend folder
csv_path = os.path.join(BASE_DIR, "shl_all_products_details.csv")
df = pd.read_csv(csv_path)

string_cols = ['name', 'link', 'Description', 'Job_levels', 'Languages', 'Assessment_length', 'Test_Type', 'Remote_Testing', 'embedding']
for col in string_cols:
    if col in df.columns:
        df[col] = df[col].fillna("")

# Now save back
df.to_csv("shl_all_products_details.csv", index=False)
# Strip whitespace from all string columns
df =df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Rename columns to match Pydantic model
df.rename(columns={
    "Job levels": "Job_levels",
    "Assessment length": "Assessment_length",
    "Test Type": "Test_Type",
    "Remote Testing": "Remote_Testing"
}, inplace=True)

# Ensure embeddings exist
if 'embedding' not in df.columns:
    print("No embeddings found. Computing embeddings now...")
    embeddings = []
    for desc in df['Description']:
        resp = client.embeddings.create(model="text-embedding-ada-002", input=str(desc))
        embeddings.append(resp.data[0].embedding)
    df['embedding'] = embeddings
    df.to_csv("shl_all_products_details.csv", index=False)
else:
    # Convert from string to list if needed
    df['embedding'] = df['embedding'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# ----------------- Initialize FastAPI -----------------
app = FastAPI(title="SHL Product MCP Server")

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------- Product Schema -----------------
class Product(BaseModel):
    name: str
    link: str
    Description: str
    Job_levels: str  # maps to "Job levels"
    Languages: str
    Assessment_length: str  # maps to "Assessment length"
    Test_Type: str  # maps to "Test Type"
    Remote_Testing: str  # maps to "Remote Testing"

# ----------------- Endpoints -----------------

@app.get("/products", response_model=list[Product])
def get_all_products():
    return df.to_dict(orient="records")

@app.get("/filter", response_model=list[Product])
def filter_products(
    job_level: Optional[str] = None,
    test_type: Optional[str] = None,
    remote: Optional[str] = None,
    language: Optional[str] = None,
    keyword: Optional[str] = None
):
    filtered = df.copy()
    
    if job_level:
        filtered = filtered[filtered['Job levels'].str.contains(job_level, case=False, na=False)]
    if test_type:
        filtered = filtered[filtered['Test Type'].str.contains(test_type, case=False, na=False)]
    if remote:
        filtered = filtered[filtered['Remote Testing'].str.lower().str.contains(remote.lower(), na=False)]
    if language:
        filtered = filtered[filtered['Languages'].str.contains(language, case=False, na=False)]
    if keyword:
        filtered = filtered[filtered['Description'].str.contains(keyword, case=False, na=False)]
    
    return filtered.to_dict(orient="records")

@app.get("/sort", response_model=list[Product])
def sort_products(
    by: str = Query("name", description="Field to sort by: name, job_level, assessment_length"),
    ascending: bool = True
):
    valid_fields = {
        "name": "name",
        "job_level": "Job levels",
        "assessment_length": "Assessment length"
    }
    field = valid_fields.get(by.lower())
    if not field:
        raise HTTPException(status_code=400, detail="Invalid sort field. Use name, job_level, or assessment_length.")
    
    sorted_df = df.sort_values(by=field, ascending=ascending)
    return sorted_df.to_dict(orient="records")

@app.get("/search", response_model=list[Product])
def search_products(keyword: str, top_n: int = 10):
    filtered = df[df['Description'].str.contains(keyword, case=False, na=False)]
    return filtered.head(top_n).to_dict(orient="records")

@app.get("/product", response_model=Product)
def get_product(name: str):
    product = df[df['name'].str.lower() == name.lower()]
    if product.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    return product.iloc[0].to_dict()

@app.get("/semantic_search", response_model=list[Product])
def semantic_search(query: str, top_n: int = 5):
    """
    Perform semantic search for multiple keywords in a single request.
    - query: comma-separated keywords, e.g., "python, seo, react"
    - top_n: number of top products to return
    """
    try:
        keywords = [q.strip() for q in query.split(",") if q.strip()]
        if not keywords:
            return []

        # Convert embeddings to numpy array
        def to_np(x):
            import numpy as np
            return np.array(x, dtype=float)

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        aggregated_scores = np.zeros(len(df))

        for kw in keywords:
            q_emb = client.embeddings.create(
                model="text-embedding-ada-002",
                input=kw
            ).data[0].embedding

            df['similarity'] = df['embedding'].apply(lambda x: cosine_sim(to_np(x), to_np(q_emb)))
            aggregated_scores += df['similarity'].values  # sum similarity scores

        # Average similarity across all keywords
        df['aggregated_similarity'] = aggregated_scores / len(keywords)

        # Return top N products
        top_products = df.sort_values(by='aggregated_similarity', ascending=False).head(top_n)

        # Drop similarity columns before returning
        return top_products.drop(columns=['similarity', 'aggregated_similarity']).to_dict(orient='records')

    except Exception as e:
        import traceback
        traceback.print_exc()
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
# ----------------- Run -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server:app", host="127.0.0.1", port=8000, reload=True)
