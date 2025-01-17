import os
import sys
from pathlib import Path
from openai import OpenAI
import argparse
from typing import Optional, List
import tiktoken
import re
try:
    from dotenv import load_dotenv
except ImportError:
    print("Please install python-dotenv: pip install python-dotenv", file=sys.stderr)
    sys.exit(1)

# Load environment variables
load_dotenv()

# Reduce max tokens to account for system prompt and response
MAX_TOKENS = 8000  # This leaves room for system prompt and response of similar length

SYSTEM_PROMPT = """Du bist ein Experte für Markdown und Daten. Deine Aufgabe ist es, Markdown-Dateien zu bereinigen und umzustrukturieren und dabei diese strengen Regeln zu befolgen:

1. Entferne alle fehlerhaften oder ungültigen Zeichen
2. Achte auf eine korrekte Markdown-Formatierung:
   - Verwende höchstens 3 Ebenen von Hashtags (###) und stelle sicher, dass die Hierarchie der Überschriften korrekt ist
   - Verwende Fett- (**) und Kursivschrift (*) für untergeordnete Überschriften (wenn nötig)
   - Achte auf einheitliche Abstände und Zeilenumbrüche
3. Richte die Hierarchie so ein, dass sie mit dem Inhaltsverzeichnis des Dokuments übereinstimmt und Sinn macht.
5. KEINE faktischen Informationen hinzufügen, entfernen oder verändern
6. NICHT halluzinieren oder neue Inhalte erzeugen
7. Behalte den gesamten Original Inhalt bei, korrigiere nur seine Formatierung und Struktur

Gebe nur den bereinigten Markdown-Inhalt ohne Erklärungen oder Kommentare aus."""

def count_tokens(text: str) -> int:
    """Count tokens in a text using tiktoken."""
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))

def split_into_chunks(content: str) -> List[str]:
    """Split content into chunks that fit within token limit."""
    # First, split content at major section boundaries (headers)
    sections = re.split(r'(?=(?:^|\n)#+ )', content)
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for section in sections:
        if not section.strip():
            continue
        
        section_tokens = count_tokens(section)
        
        # If section alone is too big, split it further at paragraphs
        if section_tokens > MAX_TOKENS:
            paragraphs = re.split(r'\n\n+', section)
            current_para_chunk = ""
            current_para_tokens = 0
            
            for para in paragraphs:
                para_tokens = count_tokens(para)
                
                # If paragraph alone is too big, split it into sentences
                if para_tokens > MAX_TOKENS:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        sentence_tokens = count_tokens(sentence)
                        if current_para_tokens + sentence_tokens > MAX_TOKENS:
                            if current_para_chunk:
                                chunks.append(current_para_chunk.strip())
                            current_para_chunk = sentence
                            current_para_tokens = sentence_tokens
                        else:
                            current_para_chunk += " " + sentence
                            current_para_tokens += sentence_tokens
                
                # Handle normal paragraphs
                elif current_para_tokens + para_tokens > MAX_TOKENS:
                    if current_para_chunk:
                        chunks.append(current_para_chunk.strip())
                    current_para_chunk = para
                    current_para_tokens = para_tokens
                else:
                    if current_para_chunk:
                        current_para_chunk += "\n\n"
                    current_para_chunk += para
                    current_para_tokens += para_tokens
            
            if current_para_chunk:
                chunks.append(current_para_chunk.strip())
        
        # If adding this section would exceed limit, start new chunk
        elif current_tokens + section_tokens > MAX_TOKENS:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
            current_tokens = section_tokens
        else:
            if current_chunk and not section.startswith('#'):
                current_chunk += '\n\n'
            current_chunk += section
            current_tokens += section_tokens
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Verify no chunk exceeds the limit
    for i, chunk in enumerate(chunks):
        chunk_tokens = count_tokens(chunk)
        if chunk_tokens > MAX_TOKENS:
            print(f"Warning: Chunk {i+1} has {chunk_tokens} tokens, which exceeds the limit of {MAX_TOKENS}")
    
    return chunks

def get_env_var(name: str, default: Optional[str] = None) -> str:
    """Get environment variable or return default."""
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Environment variable {name} is not set")
    return value

def clean_markdown_with_openai(content: str) -> str:
    """Clean up markdown content using OpenAI API."""
    client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
    
    try:
        response = client.chat.completions.create(
            model=get_env_var("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            temperature=float(get_env_var("OPENAI_TEMPERATURE", "0.1")),
            max_tokens=None if get_env_var("OPENAI_MAX_TOKENS", "None") == "None" else int(get_env_var("OPENAI_MAX_TOKENS"))
        )
        cleaned_content = response.choices[0].message.content
        if cleaned_content is None:
            raise ValueError("OpenAI API returned empty response")
        return cleaned_content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}", file=sys.stderr)
        raise

def process_file(input_path: str, output_dir: str) -> None:
    """Process a single markdown file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into chunks if needed
    total_tokens = count_tokens(content)
    if total_tokens > MAX_TOKENS:
        print(f"File is {total_tokens} tokens, splitting into chunks...")
        chunks = split_into_chunks(content)
        print(f"Split into {len(chunks)} chunks")
        
        # Process each chunk
        cleaned_chunks = []
        for i, chunk in enumerate(chunks, 1):
            chunk_tokens = count_tokens(chunk)
            print(f"Processing chunk {i}/{len(chunks)} ({chunk_tokens} tokens)...")
            cleaned_chunk = clean_markdown_with_openai(chunk)
            cleaned_chunks.append(cleaned_chunk)
        
        # Combine chunks
        cleaned_content = '\n\n'.join(cleaned_chunks)
    else:
        print(f"Processing file ({total_tokens} tokens)...")
        cleaned_content = clean_markdown_with_openai(content)
    
    # Save to output file
    output_path = os.path.join(output_dir, Path(input_path).name)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    print(f"Saved cleaned file to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Clean up markdown files using OpenAI API')
    parser.add_argument('--input', '-i', required=True, help='Input markdown file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    args = parser.parse_args()
    
    if os.path.isfile(args.input):
        # Process single file
        process_file(args.input, args.output)
    else:
        # Process directory
        for file in os.listdir(args.input):
            if file.endswith('.md'):
                input_path = os.path.join(args.input, file)
                process_file(input_path, args.output)

if __name__ == "__main__":
    main() 