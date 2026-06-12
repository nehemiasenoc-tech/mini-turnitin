import fitz  # PyMuPDF
import docx
import spacy
import re

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_sm"])
    nlp = spacy.load("es_core_news_sm")

def get_docx_text(docx_input) -> str:
    """Lee el texto de un archivo DOCX."""
    from io import BytesIO
    if isinstance(docx_input, bytes):
        docx_input = BytesIO(docx_input)
    
    doc = docx.Document(docx_input)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return " ".join(full_text)

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

def clean_text(text: str) -> list[str]:
    """Limpia y lematiza un texto."""
    spacy_doc = nlp(text)
    clean_tokens = []
    for token in spacy_doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            clean_tokens.append(token.lemma_.lower())
    return clean_tokens

def extract_and_clean_text_from_pdf(pdf_input) -> list[str]:
    """Mantiene compatibilidad con el corpus antiguo."""
    return clean_text(get_pdf_text(pdf_input))

def extract_sentences_from_text(text: str) -> list[str]:
    """Segmenta un texto en oraciones."""
    spacy_doc = nlp(text)
    return [sent.text.strip() for sent in spacy_doc.sents if len(sent.text.strip()) > 15]

def extract_sentences_from_pdf(pdf_input) -> list[str]:
    """Mantiene compatibilidad con el corpus antiguo."""
    return extract_sentences_from_text(get_pdf_text(pdf_input))

def create_chunks(sentences: list[str], chunk_size: int = 3, overlap: int = 1) -> list[str]:
    """
    Agrupa oraciones en fragmentos (chunks) para mejor análisis semántico.
    """
    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i : i + chunk_size])
        chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks
