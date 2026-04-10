import requests
from bs4 import BeautifulSoup

def get_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # ❌ supprimer scripts, styles, header, footer
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.extract()

        text = soup.get_text(separator=" ")

        # 🧹 nettoyer texte
        text = " ".join(text.split())

        return text

    except Exception as e:
        print(f"❌ Erreur scraping {url}:", e)
        return ""


def split_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)

    return chunks