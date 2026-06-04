import numpy as np
import streamlit as st

@st.cache_resource(show_spinner="Descargando y cargando modelo de IA Semántica (solo la primera vez)...")
def get_model():
    # Lazy import para evitar que la app tarde en arrancar
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def build_faiss_index(corpus_data: dict[str, dict]):
    """
    Construye un índice FAISS con todas las oraciones del corpus.
    Retorna el índice y una lista que mapea cada vector a su oración original.
    """
    import faiss
    
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
    
    return index, all_sentences

def find_suspicious_fragments(query_sentences: list[str], index, corpus_sentences: list[str], threshold: float = 0.75):
    """
    Busca fragmentos sospechosos y calcula el porcentaje general de similitud semántica.
    """
    import faiss
    
    if not query_sentences or index is None or getattr(index, 'ntotal', 0) == 0:
        return [], 0.0
        
    model = get_model()
    query_embeddings = model.encode(query_sentences, convert_to_numpy=True)
    faiss.normalize_L2(query_embeddings)
    
    # Buscamos el top 1 resultado más similar para cada oración
    D, I = index.search(query_embeddings, 1)
    
    suspicious_fragments = []
    suspicious_count = 0
    
    for i, (distances, indices) in enumerate(zip(D, I)):
        best_score = float(distances[0])
        if best_score > threshold:
            suspicious_count += 1
            matched_sentence = corpus_sentences[indices[0]]
            suspicious_fragments.append({
                'query': query_sentences[i],
                'match': matched_sentence,
                'score': best_score
            })
            
    # Calculamos el porcentaje de oraciones con similitud por encima del threshold
    semantic_similarity_percentage = (suspicious_count / len(query_sentences)) * 100 if query_sentences else 0.0
    
    # Ordenamos de mayor a menor similitud
    suspicious_fragments.sort(key=lambda x: x['score'], reverse=True)
    
    # Retornamos solo los 10 más sospechosos para no sobrecargar el prompt del agente IA
    return suspicious_fragments[:10], semantic_similarity_percentage
