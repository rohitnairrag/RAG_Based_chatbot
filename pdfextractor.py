from pypdf import PdfReader

def text_extractor_pdf(file_path):
    pdf_file=PdfReader(file_path)
    pdf_text=''
    for pages in pdf_file.pages:
        text_only=pages.extract_text()
        if text_only:
            pdf_text+=text_only
        
    return pdf_text