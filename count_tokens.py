import os
import tiktoken
from pathlib import Path

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def main():
    markdown_dir = "markdown_output"
    total_tokens = 0
    
    # Process each markdown file
    for file in sorted(os.listdir(markdown_dir)):
        if file.endswith(".md"):
            file_path = os.path.join(markdown_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            num_tokens = count_tokens(content)
            total_tokens += num_tokens
            
            print(f"{file:<30} {num_tokens:>8,} tokens")
    
    print("\n" + "="*40)
    print(f"Total tokens: {total_tokens:,}")

if __name__ == "__main__":
    main() 