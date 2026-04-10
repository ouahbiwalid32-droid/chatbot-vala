from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np
from scraper import get_website_text, split_text

app = Flask(__name__)

# =========================
# 1. Charger le modèle
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modèle chargé")

# =========================
# 2. Charger FAQ structurée
# =========================
def load_faq():
    faq_chunks = []
    try:
        with open("data/faq.txt", "r", encoding="utf-8") as f:
            contenu = f.read()

        blocs = contenu.strip().split("\n\n")
        for bloc in blocs:
            lignes = bloc.strip().split("\n")
            if len(lignes) >= 2:
                question = lignes[0].strip()
                reponse = " ".join(lignes[1:]).strip()
                faq_chunks.append(f"Q: {question}\nR: {reponse}")

        print(f"✅ {len(faq_chunks)} questions FAQ chargées")

    except Exception as e:
        print("❌ Erreur chargement FAQ:", e)

    return faq_chunks

# =========================
# 3. Scraper les sites
# =========================
def load_website():
    urls = [
        "https://www.vala-orange.com",
        "https://www.vala.ma"
    ]

    chunks = []

    for url in urls:
        try:
            print(f"🔄 Scraping: {url}")
            text = get_website_text(url)

            if text:
                site_chunks = split_text(text)
                chunks.extend(site_chunks)

        except Exception as e:
            print(f"❌ Erreur scraping {url}:", e)

    print(f"✅ {len(chunks)} chunks site web chargés")
    return chunks

# =========================
# 4. Combiner données
# =========================
faq_chunks     = load_faq()
website_chunks = load_website()

all_chunks = faq_chunks + website_chunks  # FAQ priorité

print(f"✅ Total : {len(all_chunks)} chunks combinés")

if len(all_chunks) == 0:
    raise ValueError("❌ Aucun chunk généré !")

# =========================
# 5. Embeddings + FAISS
# =========================
embeddings = model.encode(all_chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ FAISS index prêt")

# =========================
# 6. Recherche contexte
# =========================
def search_context(user_question):
    try:
        question_vector = model.encode([user_question])
        question_vector = np.array(question_vector).astype("float32")

        distances, indices = index.search(question_vector, k=5)

        context_chunks = []
        for i in indices[0]:
            if i < len(all_chunks):
                context_chunks.append(all_chunks[i])

        context = "\n".join(context_chunks)

        # DEBUG (optionnel)
        print("🧠 QUESTION:", user_question)
        print("📚 CONTEXT:", context[:300], "...")

        return context

    except Exception as e:
        print("❌ Erreur recherche:", e)
        return ""

# =========================
# 7. Appel LLM
# =========================
def ask_model(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 120  # limite réponse (plus rapide)
                }
            },
            timeout=60
        )

        data = response.json()
        return data.get("response", "Je n'ai pas pu générer une réponse.")

    except Exception as e:
        print("❌ Erreur Ollama:", e)
        return "Service temporairement indisponible. Contactez le +212 528 225 522."

# =========================
# 8. Génération réponse
# =========================
def get_response(user_question):
    try:
        context = search_context(user_question)

        prompt = f"""Tu es un assistant professionnel de Vala Orange.

Règles ABSOLUES :
- Réponds UNIQUEMENT avec les informations EXACTES du contexte
- Maximum 2-3 phrases courtes
- Si l'information n'existe pas : "Contactez-nous au +212 528 225 522."
- Pas d'invention

Contexte:
{context}

Question: {user_question}
Réponse:"""

        return ask_model(prompt)

    except Exception as e:
        print("❌ ERREUR:", e)
        return "Une erreur est survenue."

# =========================
# 9. Routes Flask
# =========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({"response": "Veuillez poser une question."})

        response = get_response(user_message)

        return jsonify({"response": response})

    except Exception as e:
        print("❌ Erreur route /chat:", e)
        return jsonify({"response": "Erreur serveur."})

# =========================
# 10. Lancer serveur
# =========================
if __name__ == "__main__":
    app.run(debug=True)