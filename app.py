from flask import Flask, request, render_template_string, jsonify
import json
import numpy as np
import faiss
from openai import OpenAI
import os

# Ensure OPENAI_API_KEY is set in Render environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load FAISS index + answers
faq_index = faiss.read_index("faq.index")
with open("answers.json", "r") as f:
    answers = json.load(f)

def query_faq(user_input, k=3, threshold=0.4):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    )
    query_vec = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_vec)
    distances, indices = faq_index.search(query_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= threshold:
            results.append(answers[idx])
    if not results:
        results = ["Sorry, I don’t know the answer yet."]
    return results

def rephrase_for_user(faq_answers, user_question):
    faq_answers = (faq_answers + ["N/A"]*3)[:3]
    prompt = f"""
FAQ Answers:
- {faq_answers[0]}
- {faq_answers[1]}
- {faq_answers[2]}

User: {user_question}
Answer naturally, clearly, and structured if needed.
"""
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful business assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content

app = Flask(__name__)

HTML_PAGE = """..."""  # Keep your HTML page here as you had it

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    matches = query_faq(user_message)
    ai_reply = rephrase_for_user(matches, user_message)
    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    app.run(debug=True)
