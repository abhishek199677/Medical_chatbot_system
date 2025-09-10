from pypdf import  PdfReader
from typing import List, Optional
from io import BytesIO


def extract_text_from_pdf(file):
    reader = PdfReader(file)  # Open the PDF file for reading
    text = ''  # Initialize an empty string to hold all text
    
    # Loop through each page in the PDF
    for page in reader.pages:
        # Extract text from the page; if None, add empty string ''
        page_text = page.extract_text() or ''
        text += page_text  # Add the page text to the overall text
    
    return text  # Return all the combined text from the PDF
