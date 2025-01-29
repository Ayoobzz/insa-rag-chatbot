import re

def clean_text(text):
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'\n+', '\n', text)  # Collapse newlines

    return text.strip()

