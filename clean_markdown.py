import os
import sys
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from openai import OpenAI, AzureOpenAI
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

# Max tokens per chunk for gpt-5-mini.
# Context window: ~400,000 tokens, max output: ~128,000 tokens.
# We keep chunks well below the context limit to leave plenty of room for
# the system prompt, any surrounding context, and the model's output.
MAX_TOKENS = 20000

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
    encoding = tiktoken.encoding_for_model("gpt-5-mini")
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


def get_llm_client_and_model():
    """Return the configured LLM client and model/deployment name.

    Provider is selected via the LLM_PROVIDER environment variable:
    - "openai" (default): uses OPENAI_* variables
    - "azure": uses AZURE_OPENAI_* variables
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "azure":
        client = AzureOpenAI(
            api_key=get_env_var("AZURE_OPENAI_API_KEY"),
            azure_endpoint=get_env_var("AZURE_OPENAI_ENDPOINT"),
            api_version=get_env_var("AZURE_OPENAI_API_VERSION"),
        )
        model = get_env_var("AZURE_OPENAI_DEPLOYMENT")
        return client, model

    if provider == "openai":
        client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
        model = get_env_var("OPENAI_MODEL")
        return client, model

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

def clean_markdown_with_openai(content: str) -> str:
    """Clean up markdown content using OpenAI API."""
    client, model = get_llm_client_and_model()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            reasoning_effort="low",
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

def process_directory(input_dir: str, output_dir: str, workers: int, executor_choice: str) -> None:
    """Process all markdown files in a directory, optionally in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    md_files = sorted(
        path for path in Path(input_dir).iterdir()
        if path.is_file() and path.suffix == '.md'
    )

    if not md_files:
        print(f"No markdown files found in {input_dir}")
        return

    total_files = len(md_files)

    if workers <= 1:
        for index, path in enumerate(md_files, 1):
            print(f"[{index}/{total_files}] Processing {path.name}")
            process_file(str(path), output_dir)
        return

    executor_cls = ProcessPoolExecutor if executor_choice == "process" else ThreadPoolExecutor
    print(f"Processing {total_files} files with {workers} {executor_choice} workers...")

    errors = []
    with executor_cls(max_workers=workers) as executor:
        future_to_path = {
            executor.submit(process_file, str(path), output_dir): path
            for path in md_files
        }

        for completed, future in enumerate(as_completed(future_to_path), 1):
            path = future_to_path[future]
            try:
                future.result()
                print(f"[{completed}/{total_files}] Finished {path.name}")
            except Exception as exc:
                errors.append((path, exc))
                print(f"[{completed}/{total_files}] Error processing {path.name}: {exc}", file=sys.stderr)

    if errors:
        print("Completed with errors:", file=sys.stderr)
        for path, exc in errors:
            print(f"- {path.name}: {exc}", file=sys.stderr)
    else:
        print("All files processed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Clean up markdown files using OpenAI API')
    parser.add_argument('--input', '-i', required=True, help='Input markdown file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of workers for directory processing. Use 1 for sequential.',
    )
    parser.add_argument(
        '--auto-workers',
        action='store_true',
        help='Automatically set workers to match file count (one worker per file).',
    )
    parser.add_argument(
        '--executor',
        choices=['process', 'thread'],
        default='process',
        help='Executor type when using multiple workers (default: process).',
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1")
    
    if os.path.isfile(args.input):
        # Process single file
        process_file(args.input, args.output)
    else:
        workers = args.workers
        if args.auto_workers:
            md_files = [
                p for p in Path(args.input).iterdir()
                if p.is_file() and p.suffix == '.md'
            ]
            workers = len(md_files) if md_files else 1
            print(f"Auto-workers: found {workers} markdown files")
        
        process_directory(args.input, args.output, workers, args.executor)

if __name__ == "__main__":
    main() 