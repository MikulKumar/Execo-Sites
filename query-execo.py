import json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI()

# Load FAISS index + answers
index = faiss.read_index("faq.index")
with open("answers.json", "r") as f:
    answers = json.load(f)

def query_faq(user_input, k=3, threshold=0.4):
    # Embed user query
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=user_input
    )
    query_vec = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

    # Normalize query for cosine similarity
    faiss.normalize_L2(query_vec)

    # Search vector DB
    distances, indices = index.search(query_vec, k)

    # Filter by similarity threshold
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= threshold:
            results.append(answers[idx])

    if not results:
        results = ["Sorry, I don’t know the answer yet."]

    # Debug: print matches
    print("Distances:", distances)
    print("Indices:", indices)

    return results

# Example usage with GPT rephrasing
if __name__ == "__main__":
    user_q = "what is the business model of your service?"
    matches = query_faq(user_q, k=3, threshold=0.4)

    # Print the matched FAQ answers
    print("Matched FAQ answers:")
    for i, match in enumerate(matches):
        print(f"{i+1}. {match}")

    # Rephrase the matched answers via GPT
    def rephrase_for_user(faq_answers, user_question):
        prompt = f"""
    You are a helpful business assistant. Use the FAQ answers below to respond naturally to the user.
    If none answer the question, say "Sorry, I don’t know."

    FAQ Answers:
    - {faq_answers[0]}
    - {faq_answers[1]}
    - {faq_answers[2]}

    User: {user_question}
    Answer:
    """
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content

    # Get the GPT-rephrased answer
    answer = rephrase_for_user(matches, user_q)
    print("\nBot response:")
    print(answer)
