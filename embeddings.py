from pypdf import PdfReader
from pathlib import Path

def read_file(path:Path):
    read = PdfReader(path)
    text = ""
    for page in read.pages:
        text += page.extract_text() + "\n"
        return text 
    
def chunk(text:str,count:int=20,overlap:int=2):
    