# main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import re

# Import your existing functions from llm_agent.py
from llm_agent import get_recommendations

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="SHL Assessment Recommender API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ----------------- Pydantic Schema -----------------
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

# ----------------- Endpoint -----------------
@app.get("/recommend", response_model=RecommendationResponse)
def recommend_assessments(query: str = Query(..., description="Job requirement or skills query")):
    try:
        recommendations = get_recommendations(query)

        # Ensure minimum 5 recommendations
        if len(recommendations) < 5:
            recommendations *= 2
        recommendations = recommendations[:10]

        # Transform into the desired schema
        result = []
        for r in recommendations:
            result.append({
                "url": r.get("url", ""),
                "name": r.get("name", ""),
                "adaptive_support": r.get("adaptive_support", ""),
                "description": r.get("description", ""),
                "duration": r.get("duration", ""),
                "remote_support": r.get("remote_support", ""),
                "test_type": r.get("test_type", "")
            })

        return {"recommended_assessments": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------- Run -----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=True)
