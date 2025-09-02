import json
import numpy as np
import faiss
from openai import OpenAI

client = OpenAI()

# Load your FAQs
with open("execo Site AI/FAQ.json", "r") as f:
    faqs = json.load(f)

# Prepare lists
texts_to_embed = []  # question + answer for embeddings
answers = []         # just the answer for retrieval

for faq in faqs:
    text_to_embed = faq["question"] + " " + faq["answer"]
    texts_to_embed.append(text_to_embed)
    answers.append(faq["answer"])  # keep answers separate

# Create embeddings
embeddings = []
for text in texts_to_embed:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embeddings.append(response.data[0].embedding)

embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)  # for cosine similarity

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# Save FAISS index and answers
faiss.write_index(index, "faq.index")
with open("answers.json", "w") as f:
    json.dump(answers, f)

print("✅ Vector DB built and saved!")
