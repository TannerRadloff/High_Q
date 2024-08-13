import os
from flask import Flask, request, jsonify, render_template
import requests
import time
from dotenv import load_dotenv
import json
import markdown
from pyngrok import ngrok, conf
from pyngrok.exception import PyngrokNgrokError
import psutil
from flask_cors import CORS
from requests.exceptions import RequestException

load_dotenv()

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load API keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")


NGROK_URL = " https://dear-driving-ewe.ngrok-free.app"


def generate_search_queries(target, num_queries=3):
    url = "https://api.together.xyz/inference"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "prompt": f"Generate {num_queries} search queries to research: {target}. Provide only the queries, one per line.",
        "max_tokens": 200,
        "temperature": 0.7
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        queries = response.json()['output']['choices'][0]['text'].strip().split('\n')
        return [query for query in queries if query.strip()][:num_queries]
    else:
        raise Exception("Failed to generate search queries")

def perform_web_search(query):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant. Provide a concise summary of the search results."},
            {"role": "user", "content": f"Perform a web search for: {query}"}
        ]
    }
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Search failed for query: {query}")


def generate_final_report(target, queries, search_results, max_retries=3):
    url = "https://api.together.xyz/inference"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    all_results = "\n\n".join([f"Query: {q}\nResult: {r}" for q, r in zip(queries, search_results)])
    prompt = f"Based on the following search results about {target}, create a comprehensive report:\n\n{all_results}\n\nProvide a well-structured report with clear sections and concise information.Use markdown formatting for headers and lists."
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "prompt": prompt,
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()['output']['choices'][0]['text']
        except RequestException as e:
            app.logger.error(f"API request failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Failed to generate final report after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

    raise Exception("Failed to generate final report due to unexpected error")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    target = data.get('target')
    if not target:
        return jsonify({"error": "No search target provided"}), 400
    
    try:
        with app.app_context():
            queries = generate_search_queries(target)
            search_results = [perform_web_search(query) for query in queries]
            report_markdown = generate_final_report(target, queries, search_results)
            report_html = markdown.markdown(report_markdown)
        return jsonify({"report": report_html})
    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        error_message = str(e) if "Failed to generate final report" in str(e) else "An unexpected error occurred"
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    print(f"Your application is accessible at: {NGROK_URL}")
    app.run(debug=True)