import os
from src.preprocess import extract_and_clean_text_from_pdf, extract_sentences_from_pdf

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
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(corpus_dir, filename)
            try:
                tokens = extract_and_clean_text_from_pdf(filepath)
                sentences = extract_sentences_from_pdf(filepath)
                corpus_data[filename] = {
                    'tokens': tokens,
                    'sentences': sentences
                }
            except Exception as e:
                print(f"Error procesando {filename} del corpus: {e}")
                
    return corpus_data
