import os
from src.preprocess import get_pdf_text, get_docx_text, clean_text, extract_sentences_from_text

def load_corpus(corpus_dir: str) -> dict[str, dict]:
    """
    Lee todos los PDFs en el directorio corpus.
    Retorna un diccionario con tokens limpios y oraciones completas:
    {
        nombre_archivo: {
            'tokens': [...],
            'sentences': [...]
        }
    }
    """
    corpus_data = {}
    if not os.path.exists(corpus_dir):
        return corpus_data
        
    for filename in os.listdir(corpus_dir):
        filepath = os.path.join(corpus_dir, filename)
        ext = filename.lower()
        try:
            raw_text = ""
            if ext.endswith(".pdf"):
                raw_text = get_pdf_text(filepath)
            elif ext.endswith(".docx"):
                raw_text = get_docx_text(filepath)
            
            if raw_text:
                tokens = clean_text(raw_text)
                sentences = extract_sentences_from_text(raw_text)
                corpus_data[filename] = {
                    'tokens': tokens,
                    'sentences': sentences
                }
        except Exception as e:
                print(f"Error procesando {filename} del corpus: {e}")
                
    return corpus_data
