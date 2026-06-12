import streamlit as st
import os
import sys

from src.preprocess import clean_text, extract_sentences_from_text, get_pdf_text, get_docx_text, create_chunks
from src.corpus import load_corpus
from src.lexical import calculate_lexical_similarity
from src.semantic import build_faiss_index, find_suspicious_fragments
from src.detector_ia import detect_ai_writing
from src.report import generate_pdf_report

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
    
    
    st.title("Mini Turnitin - Detector de Plagio")
    
    # Creación de Pestañas Principales
    tab1, tab2 = st.tabs(["🔍 Análisis de Plagio", "📚 Gestor de Corpus"])
    
    with tab1:
        st.header("Analizador de Documentos")
        uploaded_file = st.file_uploader("Sube un archivo a analizar", type=["pdf", "docx"])
        
        if uploaded_file is not None:
            max_lexical_sim = 0.0
            semantic_sim = 0.0
            most_similar_doc = "N/A"
            ai_report = {}
            suspicious_fragments = [] # FIX: Inicialización para evitar UnboundLocalError
            
            with st.spinner("Analizando documento..."):
                try:
                    file_bytes = uploaded_file.read()
                    
                    # PIPELINE: Extracción única
                    if uploaded_file.name.endswith(".pdf"):
                        raw_text = get_pdf_text(file_bytes)
                    else:
                        raw_text = get_docx_text(file_bytes)
                    
                    # Procesamiento Spacy una sola vez indirectamente
                    tokens = clean_text(raw_text)
                    query_sentences = extract_sentences_from_text(raw_text)
                    # CHUNKING: Agrupamos para el análisis semántico
                    query_chunks = create_chunks(query_sentences, chunk_size=3)
                    
                    # Carga de base de datos e índices
                    corpus_data = get_corpus_data()
                    index, corpus_sentences = get_faiss_index(corpus_data)
                    
                    # --- ANÁLISIS LÉXICO ---
                    for doc_name, doc_info in corpus_data.items():
                        # FIX: Pasamos el texto unido como string, no la lista de tokens
                        corpus_full_text = " ".join(doc_info['tokens'])
                        sim = calculate_lexical_similarity(" ".join(tokens), corpus_full_text)
                        if sim > max_lexical_sim:
                            max_lexical_sim = sim
                            most_similar_doc = doc_name
                    
                    # --- ANÁLISIS SEMÁNTICO (FAISS) ---
                    # Usamos los chunks en lugar de oraciones sueltas
                    suspicious_fragments, semantic_sim = find_suspicious_fragments(query_chunks, index, corpus_sentences)
                    
                    # --- DETECCIÓN DE ESCRITURA IA (LOCAL) ---
                    with st.spinner("Analizando patrones de escritura..."):
                        ai_report = detect_ai_writing(raw_text)
                    
                    st.success(f"**Análisis completado:** {len(tokens)} tokens y {len(query_chunks)} bloques de contexto analizados.")
                    
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar: {e}")
                
            # --- DASHBOARD DE RESULTADOS ---
            st.header("Dashboard de Análisis")
            
            # Fila de métricas principales
            m1, m2, m3 = st.columns(3)
            
            with m1:
                st.metric(
                    label="Similitud Léxica", 
                    value=f"{max_lexical_sim:.1f}%",
                    help="Mide la coincidencia exacta de palabras (tokens)."
                )
                if max_lexical_sim > 0:
                    st.caption(f"Fuente principal: **{most_similar_doc}**")
                
            with m2:
                st.metric(
                    label="Similitud Semántica", 
                    value=f"{semantic_sim:.1f}%",
                    help="Mide la coincidencia de ideas y significados mediante IA."
                )

            with m3:
                score = ai_report.get("score", 0)
                st.metric(label="Probabilidad de IA", value=f"{score}%")
            
            # Botón de Descarga de Reporte
            report_pdf = generate_pdf_report(
                uploaded_file.name,
                max_lexical_sim,
                semantic_sim,
                ai_report,
                suspicious_fragments
            )
            
            st.download_button(
                label="📥 Descargar Reporte Completo (PDF)",
                data=report_pdf,
                file_name=f"Reporte_{uploaded_file.name.split('.')[0]}.pdf",
                mime="application/pdf"
            )
            st.divider()

            # Sección de IA
            col_ia_label, col_ia_metrics = st.columns([1, 2])
            with col_ia_label:
                st.subheader("Origen del Contenido")
                color = "red" if score > 70 else "orange" if score > 40 else "green"
                st.markdown(f"""
                <div style="padding:20px; border-radius:10px; border: 2px solid {color}; text-align:center;">
                    <h3 style="color:{color}; margin:0;">{ai_report.get('label', 'Desconocido')}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col_ia_metrics:
                st.write("**Detalle de patrones lingüísticos:**")
                c1, c2, c3 = st.columns(3)
                c1.caption(f"Burstiness: {ai_report.get('burstiness',0):.2f}")
                c2.caption(f"Riqueza (TTR): {ai_report.get('ttr',0):.2f}")
                c3.caption(f"Entropía: {ai_report.get('entropy',0):.2f}")
                st.progress(score / 100)

            # --- DETALLE DE FRAGMENTOS ---
            if suspicious_fragments:
                st.subheader("🚩 Fragmentos Semánticos Detectados")
                st.write("Se han encontrado oraciones con alta similitud de ideas en la base de datos:")
                
                for idx, frag in enumerate(suspicious_fragments):
                    sim_val = frag['score'] * 100
                    card_color = "#ff4b4b" if sim_val > 85 else "#ffa500"
                    
                    with st.container(border=True):
                        st.markdown(f"**Coincidencia #{idx+1}** - Nivel de Similitud: <span style='color:{card_color}; font-weight:bold;'>{sim_val:.1f}%</span>", unsafe_allow_html=True)
                        col_doc, col_corpus = st.columns(2)
                        with col_doc:
                            st.caption("Texto en tu documento:")
                            st.info(frag['query'])
                        with col_corpus:
                            st.caption("Coincidencia en base de datos:")
                            st.success(frag['match'])
                        
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
