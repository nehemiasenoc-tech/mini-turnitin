import fitz  # PyMuPDF
import spacy
import re

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

def get_pdf_text(pdf_input) -> str:
    """Lee el texto crudo de un PDF."""
    if isinstance(pdf_input, str):
        doc = fitz.open(pdf_input)
    elif isinstance(pdf_input, bytes):
        doc = fitz.open(stream=pdf_input, filetype="pdf")
    else:
        raise ValueError("El pdf_input debe ser una ruta de archivo (str) o bytes.")

    text = ""
    for page in doc:
        text += page.get_text("text") + " "
        
    doc.close()
    return re.sub(r'\s+', ' ', text).strip()

def extract_and_clean_text_from_pdf(pdf_input) -> list[str]:
    """Extrae texto y retorna tokens limpios."""
    text = get_pdf_text(pdf_input)
    spacy_doc = nlp(text)
    
    clean_tokens = []
    for token in spacy_doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            clean_tokens.append(token.lemma_.lower())
            
    return clean_tokens

def extract_sentences_from_pdf(pdf_input) -> list[str]:
    """Extrae oraciones completas útiles para embeddings semánticos."""
    text = get_pdf_text(pdf_input)
    spacy_doc = nlp(text)
    # Filtramos oraciones muy cortas (menores a 15 caracteres)
    return [sent.text.strip() for sent in spacy_doc.sents if len(sent.text.strip()) > 15]
