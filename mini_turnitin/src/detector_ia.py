import re
import statistics
import math
from collections import Counter
from src.preprocess import nlp # Importa el modelo spacy ya cargado
from src.config import (
    AI_BURSTINESS_CRITICAL, AI_BURSTINESS_LOW,
    AI_TTR_CRITICAL, AI_TTR_LOW,
    AI_ENTROPY_CRITICAL, AI_ENTROPY_LOW,
    WEIGHT_BURSTINESS, WEIGHT_TTR, WEIGHT_ENTROPY
)

def detect_ai_writing(text: str) -> dict:
    """
    Detecta si el texto es probable que haya sido generado por IA mediante 
    análisis de patrones lingüísticos (sin APIs externas).
    """
    doc = nlp(text)

    # 1. Segmentación de oraciones usando spaCy para mayor robustez
    # Usamos los objetos 'span' de spacy para contar tokens reales
    sentences = [sent for sent in doc.sents if len(sent.text.strip()) > 15]
    
    if len(sentences) < 3:
        return {"score": 0, "label": "Texto demasiado corto", "burstiness": 0, "ttr": 0, "entropy": 0}

    # 2. Cálculo de Burstiness (Variación en la longitud de oraciones)
    # Contamos tokens que no sean puntuación para medir la estructura real
    lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
    avg_len = statistics.mean(lengths)
    # Usamos desviación estándar poblacional para evitar errores con pocas oraciones
    std_dev = statistics.pstdev(lengths)
    burstiness = std_dev / avg_len if avg_len > 0 else 0

    # 3. Cálculo de Riqueza Léxica (Type-Token Ratio)
    # La IA tiende a usar un vocabulario más "seguro" y repetitivo.
    # Usamos lemas y filtramos stopwords/puntuación para una TTR más significativa.
    words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            words.append(token.lemma_.lower())

    if not words:
        return {"score": 0, "label": "Sin contenido", "burstiness": 0, "ttr": 0, "entropy": 0}
    
    ttr = len(set(words)) / len(words)

    # 4. Cálculo de Entropía de Unigramas y Bigramas (Predictibilidad)
    word_counts = Counter(words)
    total_words = len(words)
    uni_entropy = -sum((count/total_words) * math.log2(count/total_words) for count in word_counts.values())

    # Bigramas para detectar patrones de transición (típicos de IA)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    if bigrams:
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        bi_entropy = -sum((count/total_bigrams) * math.log2(count/total_bigrams) for count in bigram_counts.values())
        # Promediamos para obtener una entropía más representativa
        entropy = (uni_entropy + bi_entropy) / 2
    else:
        entropy = uni_entropy

    # 5. Lógica de Scoring Heurística Refinada
    # Normalizamos cada métrica de 0 a 100 basado en los umbrales de config
    def normalize_score(value, critical, low, reverse=False):
        if reverse: # Para burstiness y TTR: menor valor -> más probable IA
            if value <= critical: return 100
            if value >= low: return 0
            return (low - value) / (low - critical) * 100
        else: # Para entropía: menor valor -> más probable IA
            if value <= critical: return 100
            if value >= low: return 0
            return (low - value) / (low - critical) * 100

    s_burst = normalize_score(burstiness, AI_BURSTINESS_CRITICAL, AI_BURSTINESS_LOW, reverse=True)
    s_ttr = normalize_score(ttr, AI_TTR_CRITICAL, AI_TTR_LOW, reverse=True)
    s_entropy = normalize_score(entropy, AI_ENTROPY_CRITICAL, AI_ENTROPY_LOW)

    # Aplicamos pesos configurables
    total_weight = WEIGHT_BURSTINESS + WEIGHT_TTR + WEIGHT_ENTROPY
    ai_score = (s_burst * WEIGHT_BURSTINESS + s_ttr * WEIGHT_TTR + s_entropy * WEIGHT_ENTROPY) / total_weight
    
    ai_score = min(ai_score, 100)
    
    label = "Humano"
    if ai_score > 75: label = "Altamente probable IA"
    elif ai_score > 45: label = "Posible contenido mixto/IA"
    elif ai_score > 25: label = "Probablemente Humano (con baja variedad)"

    return {
        "score": ai_score,
        "label": label,
        "burstiness": burstiness,
        "ttr": ttr,
        "entropy": entropy
    }
