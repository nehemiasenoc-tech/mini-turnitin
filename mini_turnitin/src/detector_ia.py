import asyncio
import google.generativeai as genai

async def evaluate_paraphrasing(original_text: str, suspicious_fragments: list[dict], api_key: str = "") -> tuple[str, str]:
    """
    Evalúa si existe parafraseo complejo llamando al modelo de Gemini en tiempo real.
    Retorna una tupla: (veredicto, nombre_del_modelo)
    """
    if not api_key:
        return "⚠️ **(Requiere API Key):** Por favor, ingresa tu Gemini API Key en la barra lateral izquierda para que la IA emita un veredicto.", "Ninguno"

    fragments_text = "\n".join([f"- Original: {f['match']}\n  Sospechoso: {f['query']} (Similitud Semántica: {f['score']:.2f})" for f in suspicious_fragments])
    
    prompt = f"""
    Eres un experto en detección de plagio y parafraseo complejo.
    Se te proporciona una muestra del texto de un documento subido y una lista de los fragmentos altamente sospechosos detectados mediante búsqueda semántica.
    Evalúa detalladamente si existe parafraseo complejo, alteración intencional u ocultamiento de plagio.
    Proporciona un veredicto estructurado y argumentado.
    
    Texto Subido (Muestra inicial):
    {original_text[:1500]} ...
    
    Fragmentos Altamente Sospechosos (comparados contra el corpus):
    {fragments_text}
    """
    try:
        genai.configure(api_key=api_key)
        
        # Búsqueda dinámica del modelo correcto para evitar errores 404
        valid_model_name = "gemini-1.5-flash" # Fallback por defecto
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name.lower():
                    valid_model_name = m.name
                    break
                elif 'pro' in m.name.lower():
                    valid_model_name = m.name
                    
        model = genai.GenerativeModel(valid_model_name)
        response = await model.generate_content_async(prompt)
        return response.text, valid_model_name
    except Exception as e:
        return f"❌ Error al contactar con Gemini: {str(e)}", "Error"
