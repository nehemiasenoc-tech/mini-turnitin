# --- Configuración de Umbrales de Plagio ---
# Umbral para la búsqueda inicial rápida (Bi-Encoder/FAISS)
SEMANTIC_FAISS_THRESHOLD = 0.65 
# Umbral para la validación final (Cross-Encoder). Más de 0.70 suele ser parafraseo claro.
SEMANTIC_CROSS_ENCODER_THRESHOLD = 0.75 

# --- Configuración de Detección de IA ---
AI_BURSTINESS_CRITICAL = 0.35
AI_BURSTINESS_LOW = 0.65

AI_TTR_CRITICAL = 0.38
AI_TTR_LOW = 0.48

AI_ENTROPY_CRITICAL = 4.7
AI_ENTROPY_LOW = 5.3

# Pesos para el score final de IA (0-100)
WEIGHT_BURSTINESS = 40
WEIGHT_TTR = 25
WEIGHT_ENTROPY = 35