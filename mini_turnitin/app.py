import streamlit as st
import os
import asyncio
import sys
from dotenv import load_dotenv, set_key

# Cargar variables de entorno (como el API Key guardado)
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)

# Solución al bug de asyncio en Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from src.preprocess import extract_and_clean_text_from_pdf, extract_sentences_from_pdf, get_pdf_text
from src.corpus import load_corpus
from src.lexical import calculate_lexical_similarity
from src.semantic import build_faiss_index, find_suspicious_fragments
from src.detector_ia import evaluate_paraphrasing

@st.cache_data(show_spinner="Cargando documentos base (Corpus)...")
def get_corpus_data():
    corpus_path = os.path.join("data", "corpus")
    os.makedirs(corpus_path, exist_ok=True)
    return load_corpus(corpus_path)

@st.cache_resource(show_spinner="Construyendo Índice Semántico (FAISS)...")
def get_faiss_index(_corpus_data):
    return build_faiss_index(_corpus_data)

def main():
    st.set_page_config(page_title="Mini Turnitin", layout="wide")
    
    # Barra lateral mejorada con ícono y guardado persistente
    with st.sidebar.expander("⚙️ Configuración de IA", expanded=True):
        saved_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input("🔑 Gemini API Key", type="password", value=saved_key, help="Obtén tu API Key gratis en aistudio.google.com")
        
        if st.button("💾 Guardar Clave"):
            if api_key:
                set_key(env_path, "GEMINI_API_KEY", api_key)
                st.success("¡Guardada en archivo .env local!")
            else:
                st.warning("Ingresa una clave primero.")
    
    st.title("Mini Turnitin - Detector de Plagio")
    
    # Creación de Pestañas Principales
    tab1, tab2 = st.tabs(["🔍 Análisis de Plagio", "📚 Gestor de Corpus"])
    
    with tab1:
        st.header("Analizador de Documentos")
        uploaded_file = st.file_uploader("Sube un archivo PDF a analizar", type="pdf")
        
        if uploaded_file is not None:
            max_lexical_sim = 0.0
            semantic_sim = 0.0
            most_similar_doc = "N/A"
            veredicto = ""
            modelo_usado = ""
            
            with st.spinner("Analizando documento..."):
                try:
                    pdf_bytes = uploaded_file.read()
                    
                    # Preprocesamiento
                    raw_text = get_pdf_text(pdf_bytes)
                    tokens = extract_and_clean_text_from_pdf(pdf_bytes)
                    query_sentences = extract_sentences_from_pdf(pdf_bytes)
                    
                    # Carga de base de datos e índices
                    corpus_data = get_corpus_data()
                    index, corpus_sentences = get_faiss_index(corpus_data)
                    
                    # --- ANÁLISIS LÉXICO ---
                    for doc_name, doc_info in corpus_data.items():
                        sim = calculate_lexical_similarity(tokens, doc_info['tokens'])
                        if sim > max_lexical_sim:
                            max_lexical_sim = sim
                            most_similar_doc = doc_name
                    
                    # --- ANÁLISIS SEMÁNTICO (FAISS) ---
                    suspicious_fragments, semantic_sim = find_suspicious_fragments(query_sentences, index, corpus_sentences)
                    
                    # --- AGENTE IA (GEMINI) ---
                    if suspicious_fragments:
                        with st.spinner("Consultando a Gemini para evaluación de parafraseo..."):
                            # Invocación asíncrona dentro del ciclo normal
                            veredicto, modelo_usado = asyncio.run(evaluate_paraphrasing(raw_text, suspicious_fragments, api_key))
                    else:
                        veredicto = "✅ No se encontraron fragmentos con similitud semántica suficiente para requerir evaluación profunda."
                        modelo_usado = "Ninguno"
                    
                    st.success(f"**Análisis completado:** {len(tokens)} tokens y {len(query_sentences)} oraciones procesadas.")
                    
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar: {e}")
                
            # --- DASHBOARD DE RESULTADOS ---
            st.header("Dashboard de Análisis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Similitud Léxica (Máxima)", 
                    value=f"{max_lexical_sim:.1f}%",
                    delta=f"Documento fuente: {most_similar_doc}" if max_lexical_sim > 0 else None,
                    delta_color="inverse"
                )
                
            with col2:
                st.metric(
                    label="Similitud Semántica (Global)", 
                    value=f"{semantic_sim:.1f}%",
                    delta="Basado en IA y FAISS" if semantic_sim > 0 else None,
                    delta_color="inverse"
                )
                
                st.subheader("Veredicto del Agente de IA (Parafraseo)")
                if modelo_usado and modelo_usado != "Ninguno" and modelo_usado != "Error":
                    st.caption(f"✨ **Analizado por IA:** `Modelo {modelo_usado}`")
                st.info(veredicto)
                
                # Muestra de los fragmentos sospechosos si existen
                if suspicious_fragments:
                    with st.expander("Ver detalle de fragmentos semánticamente similares"):
                        for idx, frag in enumerate(suspicious_fragments):
                            st.write(f"**{idx+1}. Similitud:** {frag['score']:.2f}")
                            st.write(f"👉 **Tu documento:** {frag['query']}")
                            st.write(f"📄 **Corpus:** {frag['match']}")
                            st.divider()
                        
    with tab2:
        st.header("Gestión de Documentos Base (Corpus)")
        st.write("Sube o elimina los documentos PDF que sirven como base de datos para comparar y detectar plagio.")
        
        corpus_path = os.path.join("data", "corpus")
        os.makedirs(corpus_path, exist_ok=True)
        
        # Panel de subida de archivos al Corpus
        st.subheader("Subir nuevos documentos")
        nuevos_archivos = st.file_uploader("Agrega PDFs a tu base de datos", type="pdf", accept_multiple_files=True, key="corpus_uploader")
        
        if st.button("Subir al Corpus", type="primary"):
            if nuevos_archivos:
                for file in nuevos_archivos:
                    file_path = os.path.join(corpus_path, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                st.success(f"✅ {len(nuevos_archivos)} archivo(s) subido(s) exitosamente a la base de datos.")
                # Forzar recarga de los datos y el índice en memoria
                get_corpus_data.clear()
                get_faiss_index.clear()
                st.rerun()
            else:
                st.warning("Selecciona al menos un archivo PDF primero.")
                
        st.divider()
        
        # Panel de visualización y borrado
        st.subheader("Documentos actuales en la base de datos")
        files = [f for f in os.listdir(corpus_path) if f.lower().endswith('.pdf')]
        
        if not files:
            st.info("La base de datos está vacía. Sube algunos PDFs para empezar.")
        else:
            for f in files:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"📄 **{f}**")
                with col2:
                    if st.button("🗑️ Eliminar", key=f"del_{f}"):
                        file_to_remove = os.path.join(corpus_path, f)
                        if os.path.exists(file_to_remove):
                            os.remove(file_to_remove)
                            st.toast(f"Se eliminó {f}")
                            # Forzar recarga tras borrar
                            get_corpus_data.clear()
                            get_faiss_index.clear()
                            st.rerun()

if __name__ == "__main__":
    main()
