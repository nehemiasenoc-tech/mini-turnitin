import numpy as np
import streamlit as st
import os
from src.config import SEMANTIC_FAISS_THRESHOLD, SEMANTIC_CROSS_ENCODER_THRESHOLD

@st.cache_resource(show_spinner="Descargando y cargando modelo de IA Semántica (solo la primera vez)...")
def get_model():
    # Lazy import para evitar que la app tarde en arrancar
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

@st.cache_resource(show_spinner="Cargando Re-ranker (Cross-Encoder)...")
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def build_faiss_index(corpus_data: dict[str, dict]):
    """
    Construye un índice FAISS con todas las oraciones del corpus.
    Retorna el índice y una lista que mapea cada vector a su oración original.
    """
    import faiss
    
    index_path = "data/faiss_index.bin"
    # Nota: Aquí faltaría lógica para verificar si el corpus cambió
    # pero como sugerencia base, la persistencia es clave.

    all_sentences = []
    for doc_info in corpus_data.values():
        all_sentences.extend(doc_info['sentences'])
        
    if not all_sentences:
        return None, []
        
    model = get_model()
    embeddings = model.encode(all_sentences, convert_to_numpy=True)
    # Normalizamos los vectores para poder usar similitud Coseno (Inner Product)
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    # faiss.write_index(index, index_path) # Ejemplo de persistencia
    
    return index, all_sentences

def find_suspicious_fragments(query_sentences: list[str], index, corpus_sentences: list[str]):
    """
    Busca fragmentos sospechosos y calcula el porcentaje general de similitud semántica.
    """
    import faiss
    
    if not query_sentences or index is None or getattr(index, 'ntotal', 0) == 0:
        return [], 0.0
        
    cross_encoder = get_cross_encoder()
    model = get_model()
    query_embeddings = model.encode(query_sentences, convert_to_numpy=True)
    faiss.normalize_L2(query_embeddings)
    
    # Buscamos el top 5 resultados más similares para tener mejores candidatos para el Cross-Encoder
    k_candidates = 5
    D, I = index.search(query_embeddings, k_candidates)
    
    suspicious_fragments = []
    suspicious_count = 0
    
    for i, (distances, indices) in enumerate(zip(D, I)):
        # Filtrar candidatos iniciales con el umbral de FAISS
        candidates = [
            {'query': query_sentences[i], 'match': corpus_sentences[idx], 'bi_score': float(dist)}
            for dist, idx in zip(distances, indices) if dist > SEMANTIC_FAISS_THRESHOLD
        ]
        
        if not candidates:
            continue

        # Re-ranking con Cross-Encoder para estos candidatos
        pairs = [[c['query'], c['match']] for c in candidates]
        # Aplicamos función Sigmoide para normalizar los logits al rango [0, 1]
        raw_scores = cross_encoder.predict(pairs)
        normalized_scores = 1 / (1 + np.exp(-raw_scores))
        
        # Buscamos si alguno supera el umbral definitivo
        for idx, score in enumerate(normalized_scores):
            if score > SEMANTIC_CROSS_ENCODER_THRESHOLD:
                suspicious_count += 1
                suspicious_fragments.append({
                    'query': candidates[idx]['query'],
                    'match': candidates[idx]['match'],
                    'score': float(score)
                })
                break # Solo contamos la mejor coincidencia por fragmento
            
    # Calculamos el porcentaje de oraciones con similitud por encima del threshold
    semantic_similarity_percentage = (suspicious_count / len(query_sentences)) * 100 if query_sentences else 0.0
    
    suspicious_fragments.sort(key=lambda x: x['score'], reverse=True)
    # Retornamos solo los 10 más sospechosos para no sobrecargar el prompt del agente IA
    return suspicious_fragments[:10], semantic_similarity_percentage
