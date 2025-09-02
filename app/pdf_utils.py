from pypdf import  PdfReader
from typing import List, Optional
from io import BytesIO


# Extract text from a PDF file
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ' '
    for page in reader.pages:
        text = text + page.extract_text() or ''
    return text