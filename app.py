from flask import Flask, request, jsonify, render_template, Response
from sentence_transformers import SentenceTransformer
import requests
import faiss
import numpy as np
from scraper import get_website_text, split_text
import sqlite3
from datetime import datetime
from functools import wraps
from time import time

app = Flask(__name__)


# ======================
# Base de données logs
# ======================
def init_logs_db():
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        session_id TEXT,
        question TEXT,
        reponse TEXT,
        temps_reponse REAL
    )''')
    conn.commit()
    conn.close()

init_logs_db()


def save_log(session_id, question, reponse, temps_reponse):
    try:
        conn = sqlite3.connect('chatbot_logs.db')
        c = conn.cursor()
        c.execute('''INSERT INTO logs (timestamp, session_id, question, reponse, temps_reponse)
                     VALUES (?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), session_id, question, reponse, temps_reponse))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Erreur sauvegarde log:", e)


# ======================
# Authentification admin
# ======================
def check_auth(username, password):
    return username == 'admin' and password == 'admin123'

def authenticate():
    return Response('Accès refusé', 401, {'WWW-Authenticate': 'Basic realm="Login required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


# ======================
# Chargement modèle + FAQ + scraping
# ======================
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Modèle chargé")


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


faq_chunks = load_faq()
website_chunks = load_website()
all_chunks = faq_chunks + website_chunks

print(f"📦 Total : {len(all_chunks)} chunks combinés")

if len(all_chunks) == 0:
    raise ValueError("❌ Aucun chunk généré !")


# ======================
# FAISS index
# ======================
embeddings = model.encode(all_chunks)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ FAISS index prêt")


# ======================
# Recherche contexte
# ======================
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

        print("💬 QUESTION:", user_question)
        print("📖 CONTEXT:", context[:300], "...")

        return context

    except Exception as e:
        print("❌ Erreur recherche:", e)
        return ""


# ======================
# Appel LLM (Ollama)
# ======================
def ask_model(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 120
                }
            },
            timeout=60
        )

        data = response.json()
        return data.get("response", "Je n'ai pas pu générer une réponse.")

    except Exception as e:
        print("❌ Erreur Ollama:", e)
        return "Service temporairement indisponible. Contactez le +212 528 225 522."


# ======================
# Génération réponse
# ======================
def get_response(user_question, session_id="default"):
    try:
        context = search_context(user_question)

        prompt = f"""Tu es un assistant professionnel de Vala Orange.

OBJECTIF :
Répondre à la question de l'utilisateur en utilisant uniquement les informations fournies dans le contexte.

RÈGLES STRICTES :
- Réponds directement à la question, sans introduction inutile
- Ne parle jamais de toi (pas de "je suis un assistant")
- N'invente aucune information
- N'utilise que les données présentes dans le contexte
- Réponse courte et professionnelle (maximum 2 phrases)
- Si l'information n'existe pas dans le contexte, réponds EXACTEMENT :
"Contactez-nous au +212 528 225 522."

STYLE :
- Clair
- Précis
- Ton professionnel (support client)

Contexte:
{context}

Question: {user_question}
Réponse:"""

        return ask_model(prompt)

    except Exception as e:
        print("❌ ERREUR:", e)
        return "Une erreur est survenue."


# ======================
# Routes Flask
# ======================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")

        if not user_message.strip():
            return jsonify({"response": "Veuillez poser une question."})

        debut = time()
        response = get_response(user_message, session_id)
        fin = time()
        temps_reponse = round(fin - debut, 3)

        save_log(session_id, user_message, response, temps_reponse)

        return jsonify({"response": response})

    except Exception as e:
        print("❌ Erreur route /chat:", e)
        return jsonify({"response": "Erreur serveur."})


@app.route("/admin")
@requires_auth
def admin():
    conn = sqlite3.connect('chatbot_logs.db')
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM logs")
    total_questions = c.fetchone()[0]

    c.execute("SELECT AVG(temps_reponse) FROM logs")
    temps_moyen = round(c.fetchone()[0] or 0, 2)

    c.execute("SELECT question, COUNT(*) as nb FROM logs GROUP BY question ORDER BY nb DESC LIMIT 5")
    top_questions = c.fetchall()

    # Questions par jour (7 derniers jours)
    c.execute("""SELECT DATE(timestamp), COUNT(*) FROM logs 
                 WHERE timestamp >= DATE('now', '-7 days') 
                 GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)""")
    questions_par_jour = c.fetchall()
    jours = [q[0] for q in questions_par_jour]
    nb_questions_par_jour = [q[1] for q in questions_par_jour]

    # Temps moyen par jour
    c.execute("""SELECT DATE(timestamp), AVG(temps_reponse) FROM logs 
                 WHERE timestamp >= DATE('now', '-7 days') 
                 GROUP BY DATE(timestamp) ORDER BY DATE(timestamp)""")
    temps_par_jour = c.fetchall()
    temps_moyens_par_jour = [round(t[1] or 0, 2) for t in temps_par_jour]

    # Derniers logs
    c.execute("SELECT timestamp, session_id, question, reponse, temps_reponse FROM logs ORDER BY timestamp DESC LIMIT 100")
    logs = c.fetchall()
    
    # ======================
# 🔥 Taux réussite / échec
# ======================
    total = len(logs)

    fail_count = sum(1 for log in logs if "Contactez-nous" in log[3])
    success_count = total - fail_count

    success_rate = round((success_count / total) * 100, 2) if total > 0 else 0
    fail_rate = round((fail_count / total) * 100, 2) if total > 0 else 0
    
    # ======================
# 🔥 Questions non comprises
# ======================
    failed_questions = [log[2] for log in logs if "Contactez-nous" in log[3]]

    conn.close()

    return render_template(
        "admin.html",
        total_questions=total_questions,
        temps_moyen=temps_moyen,
        top_questions=top_questions,
        jours=jours,
        nb_questions_par_jour=nb_questions_par_jour,
        temps_moyens_par_jour=temps_moyens_par_jour,
        logs=logs,

        # 🔥 nouveaux
        success_rate=success_rate,
        fail_rate=fail_rate,
        failed_questions = list(set(
            log[2] for log in logs if "Contactez-nous" in log[3]
        ))
    )
 

def reload_faiss():
    global faq_chunks, website_chunks, all_chunks
    global embeddings, index

    print("🔄 Rechargement FAISS...")

    faq_chunks = load_faq()
    website_chunks = load_website()
    all_chunks = faq_chunks + website_chunks

    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("✅ FAISS mis à jour")
    
@app.route("/add_faq", methods=["POST"])
@requires_auth
def add_faq():
    try:
        data = request.get_json()
        question = data.get("question")
        reponse = data.get("reponse")

        if not question or not reponse:
            return jsonify({"message": "Données invalides"})

        with open("data/faq.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n{question}\n{reponse}")
        
        reload_faiss()

        return jsonify({"message": "Ajouté au FAQ avec succès ✅"})

    except Exception as e:
        print("❌ Erreur ajout FAQ:", e)
        return jsonify({"message": "Erreur serveur"})


# ======================
# Lancer serveur
# ======================
if __name__ == "__main__":
    app.run(debug=True)