from flask import Flask, request, render_template_string, jsonify
import json
from dotenv import load_dotenv
load_dotenv() 
import numpy as np
import faiss
from openai import OpenAI
import os 

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
    # Ensure there are at least 3 items
    faq_answers = (faq_answers + ["N/A"]*3)[:3]

    prompt = f"""
    company info: 
        company name: Execo Site
        value: we get you more customers, not just a website
        Execo Sites does not only build websites, it manage your whole online presence for you
        we aren’t one of those agencies who make the website for you and just leave. you at it to figure it out, we guide you with AI and technology.
        We host and mange the websites for you
        Websites create the foundation of trust and credibility, it shows that this person knows what he is doing.
        You can get customers at any time of the day, so you dont miss out on any clients
        A well setup site can help you appear higher on the google page, which attracts more cusomters
        We also provide you with insights on your website and how customers are liking the new website
        A website is basically a low cost ad, you just have to set it up once and then it basically works for you.

    FAQ Answers:
    - {faq_answers[0]}
    - {faq_answers[1]}
    - {faq_answers[2]}

    User: {user_question}
    You are named Exie. Answer naturally and clearly in a **structured format with headings, bullet points, and bold plan names** if they ask about any info that needs to be structured, dont structure if it is a normal conversation.
    If none answer the question, say "Sorry, I don’t know."
    """

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful business assistant. Always respond in a clear, structured, bullet-point format. Use headings, bold plan names, and list features under each plan."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content


# ------------------ Flask App ------------------ #
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Execo AI Chat</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            font-family: Arial, sans-serif;
            background-color: #f0f2f5;
            margin: 0;
        }
        #chat-container {
            background: white;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            width: 500px;
        }
        #chat-box {
            height: 350px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        input[type="text"] {
            width: 75%;
            padding: 8px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
        button {
            padding: 8px 12px;
            border: none;
            border-radius: 5px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
        }
        button:hover { background-color: #0056b3; }
        #chat-box h1, h2, h3, h4 { margin: 5px 0; }
        #chat-box ul { padding-left: 20px; }
        #chat-box li { margin-bottom: 5px; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div id="chat-container">
        <h2 style="text-align:center;">Execo AI Chat</h2>
        <div id="chat-box"></div>
        <form id="chat-form">
            <input type="text" id="user-input" placeholder="Type your message" autocomplete="off">
            <button type="submit">Send</button>
        </form>
    </div>

    <script>
        const form = document.getElementById('chat-form');
        const input = document.getElementById('user-input');
        const chatBox = document.getElementById('chat-box');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const userMessage = input.value.trim();
            if(!userMessage) return;
            chatBox.innerHTML += "<p><b>You:</b> " + userMessage + "</p>";
            input.value = "";
            chatBox.scrollTop = chatBox.scrollHeight;

            const response = await fetch('/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: userMessage})
            });

            const data = await response.json();
            // Render GPT Markdown properly
            chatBox.innerHTML += "<p><b>Exie:</b></p>" + marked.parse(data.reply);
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/ask", methods=["POST"])
def ask():
    user_message = request.json.get("message")
    matches = query_faq(user_message, k=3, threshold=0.4)
    ai_reply = rephrase_for_user(matches, user_message)
    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    app.run(debug=True)
