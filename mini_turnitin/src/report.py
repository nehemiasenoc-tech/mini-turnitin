from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        # Encabezado del reporte
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'Reporte de Originalidad - Mini Turnitin', border=False, ln=1, align='C')
        self.set_draw_color(0, 80, 180)
        self.line(10, 22, 200, 22)
        self.ln(10)

    def _safe_multi_cell(self, w, h, txt, border=0, ln=0, align='J', fill=False, link=''):
        """
        Wrapper around multi_cell to handle extremely long words that FPDF
        might not be able to break, causing "Not enough horizontal space" error.
        """
        max_word_length = 50 # Adjust as needed. A URL or hash could be longer.
        processed_text_lines = []
        for line in txt.split('\n'): # Handle existing newlines
            processed_text_lines.append(break_long_words(line, max_word_length))
        self.multi_cell(w, h, safe_text('\n'.join(processed_text_lines)), border, ln, align, fill, link)
        
    def footer(self):
        # Pie de página con numeración
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', align='C')

def break_long_words(text, max_word_length=50):
    """
    Inserts spaces into very long words to prevent FPDF from failing
    when a single 'word' exceeds the available line width and cannot be broken.
    """
    words = text.split(' ')
    broken_words = []
    for word in words:
        if len(word) > max_word_length:
            broken_word = ' '.join([word[i:i+max_word_length] for i in range(0, len(word), max_word_length)])
            broken_words.append(broken_word)
        else:
            broken_words.append(word)
    return ' '.join(broken_words)

def safe_text(text):
    """Limpia el texto para asegurar compatibilidad con la codificación Latin-1 de FPDF."""
    return str(text).encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(filename, lexical_sim, semantic_sim, ai_report, fragments):
    """
    Genera un archivo PDF con el resumen detallado del análisis.
    """
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # 1. Información General
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, safe_text(f"Analisis del Documento: {filename}"), ln=1)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 7, safe_text(f"Fecha de emision: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}"), ln=1)
    pdf.ln(5)

    # 2. Resumen de Metricas (Cuadro de Honor)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, safe_text("Resumen de Resultados"), ln=1, fill=True)
    pdf.set_font('helvetica', '', 11)
    pdf.cell(0, 8, safe_text(f"  > Similitud Lexica (Palabras exactas): {lexical_sim:.1f}%"), ln=1)
    pdf.cell(0, 8, safe_text(f"  > Similitud Semantica (Ideas/Parafraseo): {semantic_sim:.1f}%"), ln=1)
    pdf.cell(0, 8, safe_text(f"  > Probabilidad de Contenido IA: {ai_report.get('score', 0):.1f}%"), ln=1)
    pdf.cell(0, 8, safe_text(f"  > Veredicto: {ai_report.get('label', 'N/A')}"), ln=1)
    pdf.ln(5)

    # 3. Detalle Lingüístico (IA)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, safe_text("Indicadores de Escritura IA"), ln=1)
    pdf.set_font('helvetica', '', 10)
    pdf.cell(0, 7, safe_text(f"- Burstiness (Variacion de ritmo): {ai_report.get('burstiness', 0):.2f}"), ln=1)
    pdf.cell(0, 7, safe_text(f"- Riqueza Lexica (TTR): {ai_report.get('ttr', 0):.2f}"), ln=1)
    pdf.cell(0, 7, safe_text(f"- Entropia (Predictibilidad): {ai_report.get('entropy', 0):.2f}"), ln=1)
    pdf.ln(10)

    # 4. Fragmentos Sospechosos
    if fragments:
        pdf.set_font('helvetica', 'B', 12)
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 10, safe_text("Detalle de Fragmentos Sospechosos"), ln=1)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)
        
        for i, frag in enumerate(fragments):
            # Salto de página si no hay espacio para la siguiente coincidencia
            if pdf.get_y() > 220:
                pdf.add_page()
            
            score_pct = frag['score'] * 100
            pdf.set_font('helvetica', 'B', 10)
            pdf.cell(0, 8, safe_text(f"Coincidencia #{i+1} - Nivel de Similitud: {score_pct:.1f}%"), ln=1)
            
            pdf.set_font('helvetica', 'I', 9)
            pdf._safe_multi_cell(0, 5, f"Texto analizado: {frag['query']}")
            pdf.set_font('helvetica', '', 9)
            pdf.set_text_color(50, 50, 50)
            pdf._safe_multi_cell(0, 5, f"Fuente encontrada: {frag['match']}")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

    # Convertimos bytearray a bytes para compatibilidad con Streamlit
    return bytes(pdf.output())