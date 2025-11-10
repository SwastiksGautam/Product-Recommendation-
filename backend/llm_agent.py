import requests
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup
from trafilatura import extract as trafilatura_extract

load_dotenv()

# Access your key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MCP_SERVER_URL = "http://127.0.0.1:10000"


# ----------------- Step 0: Extract text from URL -----------------
def extract_text_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return ""
        html = resp.text
        text = trafilatura_extract(html)
        if text:
            return text
        soup = BeautifulSoup(html, "html.parser")
        candidates = soup.find_all(["p","div","span"])
        text_content = " ".join([c.get_text(separator=" ").strip() for c in candidates])
        return re.sub(r"\s+", " ", text_content)
    except Exception as e:
        print("Error fetching URL:", e)
        return ""


# ----------------- Utility -----------------
def robust_json_parse(llm_output: str) -> dict:
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}|\[.*\]', llm_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {}

# ----------------- MCP Server Tool Wrappers -----------------
def semantic_search(keywords, top_n=10):
    try:
        resp = requests.get(f"{MCP_SERVER_URL}/semantic_search", params={"query": ",".join(keywords), "top_n": top_n})
        return resp.json()
    except:
        return []

def exact_search(keyword):
    try:
        resp = requests.get(f"{MCP_SERVER_URL}/search", params={"keyword": keyword})
        return resp.json()
    except:
        return []

def filter_products(job_level=None, test_type=None, remote=None, language=None, keyword=None):
    try:
        params = {k:v for k,v in {
            "job_level": job_level,
            "test_type": test_type,
            "remote": remote,
            "language": language,
            "keyword": keyword
        }.items() if v}
        resp = requests.get(f"{MCP_SERVER_URL}/filter", params=params)
        return resp.json()
    except:
        return []

def sort_products(by="name", ascending=True):
    try:
        resp = requests.get(f"{MCP_SERVER_URL}/sort", params={"by": by, "ascending": str(ascending)})
        return resp.json()
    except:
        return []

# ----------------- Step 1: Extract Intent -----------------
def extract_intent(user_query: str) -> dict:
    prompt = f"""
You are an expert job-product recommender. 
Extract the following fields strictly as JSON:
- skill
- job_level
- remote
- test_type
- language (optional)
- assessment_length (optional)

Example JSON:
{{
    "skill": ".NET",
    "job_level": "Mid-Professional",
    "remote": "Yes",
    "test_type": "Cognitive",
    "language": "English",
    "assessment_length": "10"
}}

User query: "{user_query}"
Return only valid JSON.
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are precise and output strictly valid JSON."},
            {"role": "user", "content": prompt}
        ]
    )
    return robust_json_parse(resp.choices[0].message.content)


# ----------------- Step 2: Sort by duration helper -----------------
def sort_by_duration(candidates, target_duration):
    if not target_duration:
        return candidates
    def duration_diff(p):
        try:
            return abs(int(p.get("duration", 0)) - target_duration)
        except:
            return float('inf')
    return sorted(candidates, key=duration_diff)


# ----------------- Step 2: LLM-powered Autonomous Ranking -----------------
def llm_tool_autonomous_recommendation(user_query, intent, initial_candidates, max_iterations=3):
    candidate_products = initial_candidates
    target_duration = None
    if intent.get("assessment_length"):
        try:
            target_duration = int(intent["assessment_length"])
        except:
            pass

    # Sort initially by duration
    candidate_products = sort_by_duration(candidate_products, target_duration)

    tools = {
        "semantic_search": semantic_search,
        "exact_search": exact_search,
        "filter_products": filter_products,
        "sort_products": sort_products
    }

    for iteration in range(max_iterations):
        prompt = f"""
You are a top-tier job-product sales AI.

User query: "{user_query}"

Structured intent:
{json.dumps(intent, indent=2)}

Candidate products (iteration {iteration+1}):
{json.dumps(candidate_products, indent=2)}

You have access to these Python functions as tools:
1. semantic_search(keywords, top_n)
2. exact_search(keyword)
3. filter_products(job_level, test_type, remote, language, keyword)
4. sort_products(by, ascending)

Instructions:
- Evaluate each candidate product for relevance to user intent.
- Rank products by relevance and duration closeness.
- Discard irrelevant products.
- Rank remaining products by relevance.
- You may loop internally and call tools multiple times.
- Return strictly valid JSON list of 5 to 10 products:
[ 
  {{"url": "...", "name": "...", "adaptive_support": "...", "description": "...", "duration": "...", "remote_support": "...", "test_type": "..."}}
]
- Do NOT include text outside JSON.
- You may choose not to use all tools if unnecessary.
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are precise and output strictly valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        refined = robust_json_parse(resp.choices[0].message.content)
        if refined:
            if isinstance(refined, dict):
                refined = [refined]
            elif not isinstance(refined, list):
                refined = list(refined)

        # Sort by duration again
        refined = sort_by_duration(refined, target_duration)

        # Ensure at least 5
        if refined and len(refined) >= 5:
            return refined[:10]  # max 10

        candidate_products = refined if refined else candidate_products

    # If after all iterations we still have <5, duplicate existing to reach min 5
    if not isinstance(candidate_products, list):
        candidate_products = list(candidate_products)
    while len(candidate_products) < 5 and candidate_products:
        candidate_products.append(candidate_products[len(candidate_products) % len(candidate_products)])

    return candidate_products[:10]  # max 10



# ----------------- Step 3: Main Workflow -----------------
def get_recommendations(user_input: str, input_type: str = "text"):
    """
    input_type: "text" (natural language), "url" (webpage URL)
    """
    # If URL, extract text first
    if input_type == "url":
        user_query = extract_text_from_url(user_input)
        if not user_query:
            return []
    else:
        user_query = user_input

    intent = extract_intent(user_query)

    target_duration = None
    if intent.get("assessment_length"):
        try:
            target_duration = int(intent["assessment_length"])
        except:
            pass

    keywords = [s.strip() for s in re.split(r',|\band\b', intent.get("skill",""), flags=re.IGNORECASE) if s.strip()]

    candidate_products = semantic_search(keywords)
    if not candidate_products:
        candidate_products = []
        for kw in keywords:
            candidate_products.extend(exact_search(kw))
        candidate_products = list({p['name']:p for p in candidate_products}.values())

    if not candidate_products:
        return []

    # Sort initially by duration
    candidate_products = sort_by_duration(candidate_products, target_duration)

    recommended_products = llm_tool_autonomous_recommendation(user_query, intent, candidate_products)
    return recommended_products
# ----------------- Step 2a: Clean Recommendation -----------------
def clean_recommendation(rec):
    """Ensure output matches the SHL Assessment Resource Schema."""

    # duration as integer
    dur = rec.get("duration")
    if isinstance(dur, str):
        match = re.search(r"\d+", dur)
        dur = int(match.group()) if match else None
    elif isinstance(dur, (int, float)):
        dur = int(dur)
    else:
        dur = None

    # Full assessment category mapping
    assessment_map = {
        "A": "Ability & Aptitude",
        "B": "Biodata & Situational Judgement",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Assessment Exercises",
        "K": "Knowledge & Skills",
        "P": "Personality & Behaviour",
        "S": "Simulations"
    }

    # test_type mapping
    tt = rec.get("test_type", "")
    tt_list = []

    if isinstance(tt, str):
        for t in tt.split(","):
            t = t.strip().upper()
            tt_list.append(assessment_map.get(t, t) if t else "")
    elif isinstance(tt, list):
        for t in tt:
            t = str(t).strip().upper()
            tt_list.append(assessment_map.get(t, t))
    
    # remove empty strings
    tt_list = [x for x in tt_list if x]

    return {
        "url": rec.get("url", ""),
        "name": rec.get("name", ""),
        "adaptive_support": "Yes" if str(rec.get("adaptive_support", "")).lower() == "yes" else "No",
        "description": rec.get("description", ""),
        "duration": dur,  # always integer
        "remote_support": "Yes" if str(rec.get("remote_support", "")).lower() == "yes" else "No",
        "test_type": tt_list  # always list of strings
    }
