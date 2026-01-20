import os
from pathlib import Path
import pdf4llm
import re

def clean_text(text: str) -> str:
    """Clean up problematic characters and normalize text."""
    # Replace problematic characters (like multiple dots) with proper characters
    text = re.sub('\u2026', '...', text)  # Replace ellipsis
    text = re.sub('\u00A0', ' ', text)    # Replace non-breaking spaces
    text = re.sub(r'\.{2,}', ' . . . ', text)  # Format multiple dots properly
    # Remove any remaining invalid characters
    text = ''.join(char for char in text if ord(char) < 65536 and char.isprintable())
    return text

def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> None:
    """
    Convert a PDF file to markdown format using PyMuPDF4LLM.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the markdown output
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the PDF filename without extension
    pdf_name = Path(pdf_path).stem
    
    # Process the PDF and get markdown content
    markdown_content = pdf4llm.to_markdown(pdf_path)
    
    # Clean up the text
    markdown_content = clean_text(markdown_content)
    
    # Save to markdown file with explicit UTF-8 encoding
    output_path = os.path.join(output_dir, f"{pdf_name}.md")
    with open(output_path, "w", encoding="utf-8", errors="replace") as f:
        f.write(markdown_content)
    print(f"Converted {pdf_path} to {output_path}")

def main():
    # Define input and output directories
    data_dir = "data/original"
    output_dir = "markdown_output"
    
    # Process all PDFs in the data directory
    for file in os.listdir(data_dir):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file)
            convert_pdf_to_markdown(pdf_path, output_dir)

if __name__ == "__main__":
    main()
