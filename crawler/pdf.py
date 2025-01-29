import pdfplumber


def extract_pdf_text(pdf_url):
    with pdfplumber.open(pdf_url) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages])
    return text
