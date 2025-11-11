SHL Product Recommendation System
Overview
This project is an intelligent recommendation system designed to help hiring managers and recruiters find the right SHL assessments for their job roles. The system addresses the inefficiency of traditional keyword searches by accepting a natural language query or a full job description (JD)  and returning a list of the most relevant assessments.



The core of the system is a Large Language Model (LLM) that functions as a reasoning agent. Instead of just matching keywords, the LLM analyzes the user's intent, job level, skill requirements, and other constraints. It then uses a set of tools to search, filter, and rank products, mimicking the consultative process of a human salesperson to provide highly accurate recommendations. 





Architecture
The system is comprised of two main components:

Backend (FastAPI on Render):

A Python-based API built with FastAPI.

Serves all API endpoints for search, filtering, and recommendation.

Manages loading product data and embeddings from shl_all_products_details.csv.

Hosts the llm_agent logic that orchestrates the recommendation process.

Frontend (JavaScript on Vercel):

A simple HTML/CSS/JavaScript client.

Provides a user interface with a text box to enter a query.

Calls the backend's /recommend endpoint on the Render production URL.

Renders the returned list of recommended assessments.

Core Logic: LLM as Reasoning Agent
The system's logic is based on Solution Approach 2 from the project report, which treats the LLM as a reasoning agent with tool access. This was chosen over a simpler keyword-based vector search, which was found to lose user intent and context. 



The recommendation process works as follows:


Query Input: The user provides a query (e.g., "Frontend developer with SEO experience" ) or a URL to a job description. If a URL is provided, the text is extracted.

Intent Extraction: The query is first sent to an LLM (extract_intent) to parse it into a structured JSON object. This identifies key fields like skill, job_level, remote, and test_type. 

Candidate Generation: The system generates an initial list of candidate products. It first attempts a semantic_search using the extracted skills. If that yields no results, it falls back to an exact_search for each keyword.

LLM Agent Reasoning: The core function llm_tool_autonomous_recommendation is triggered. The LLM (e.g., GPT-4o-mini) is given:

The original user query. 

The structured intent (from step 2). 

The list of candidate products (from step 3). 

Access to tools like filter_products and sort_products. 


Iterative Refinement: The LLM evaluates the candidates against the intent. It logically filters out irrelevant products (e.g., wrong job level, missing remote support) and re-ranks the relevant ones. This step mimics a human consultant and ensures constraints are respected. 





Final Output: The agent returns a refined, deduplicated, and ranked list of the top 5-10 assessments. The backend then formats this list to match the strict RecommendationResponse schema before sending it to the user.

Local Setup and Running (Backend)
To run the FastAPI server locally:

Clone the repository:

Bash

git clone <repository-url>
cd <repository-directory>
Create a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies: (Create a requirements.txt file with the following content and run pip install -r requirements.txt)

fastapi
uvicorn[standard]
pandas
numpy
openai
python-dotenv
requests
beautifulsoup4
trafilatura
Create .env file: Create a file named .env in the root directory and add your OpenAI API key:

OPENAI_API_KEY="sk-..."
Add Data File: Place the shl_all_products_details.csv file in the same directory as app.py. The server will generate embeddings on first run if they are not present in the CSV.

Run the server:

Bash

uvicorn app:app --host 0.0.0.0 --port 10000 --reload
The API will be available at http://127.0.0.1:10000.

API Endpoints
The FastAPI server (app.py) exposes the following endpoints:

POST /recommend

The main endpoint for getting recommendations.

Body: {"query": "...", "input_type": "text" | "url"}

Response: RecommendationResponse object containing a list of AssessmentResource items.

GET /health

Health check endpoint.

Response: {"status": "healthy"}

GET /products

Returns a list of all products from the CSV.

Response: List[Product]

GET /filter

Filters products based on query parameters.

Query Params: job_level (str), test_type (str), remote (str), language (str), keyword (str)

Response: List[Product]

GET /sort

Sorts all products by a specific field.

Query Params: by (str: "name", "job_level", "assessment_length"), ascending (bool)

Response: List[Product]

GET /semantic_search

Performs semantic vector search on product embeddings.

Query Params: query (str: comma-separated keywords), top_n (int)

Response: List[Product]
