from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_lexical_similarity(text1: str, text2: str) -> float:
    """
    Calcula la similitud usando TF-IDF y Coseno, que es más preciso que Jaccard.
    """
    if not text1.strip() or not text2.strip():
        return 0.0
    
    vectorizer = TfidfVectorizer()
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100)
    except:
        return 0.0
