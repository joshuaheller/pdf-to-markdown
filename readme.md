# Project Overview

This project involves processing PDF files to convert them into Markdown format and then cleaning up the Markdown files using the OpenAI API. The project is structured into two main scripts: `pdf_to_markdown.py` and `clean_markdown.py`.

## Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`):
  - PyMuPDF4LLM
  - tiktoken
  - openai
  - python-dotenv

## Installation

1. Clone the repository.
2. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

Ensure you have a `.env` file in the root directory with the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_MODEL`: The model to use for the OpenAI API.
- `OPENAI_TEMPERATURE`: (Optional) The temperature setting for the OpenAI API.
- `OPENAI_MAX_TOKENS`: (Optional) The maximum number of tokens for the OpenAI API.

## Scripts and Commands

### 1. Convert PDF to Markdown

The `pdf_to_markdown.py` script processes PDF files and converts them into Markdown format.

#### Usage

```bash
python pdf_to_markdown.py
```

- **Input Directory**: `data` (default directory where PDF files are stored)
- **Output Directory**: `markdown_output` (default directory where converted Markdown files are saved)

### 2. Clean Markdown Files

The `clean_markdown.py` script cleans and restructures the Markdown files using the OpenAI API.

#### Usage

```bash
python clean_markdown.py --input <input_path> --output <output_dir>
```

- **Input Path**: Path to a single Markdown file or a directory containing Markdown files.
- **Output Directory**: Directory where cleaned Markdown files will be saved.

### 3. Count Tokens in Markdown Files

The `count_tokens.py` script calculates the number of tokens in each Markdown file within a specified directory using the `tiktoken` library.

#### Usage

```bash
python count_tokens.py
```

- **Markdown Directory**: `markdown_output` (default directory where Markdown files are stored)
- The script outputs the number of tokens for each file and the total number of tokens across all files in the directory.

## Notes

- Ensure that the OpenAI API key and other environment variables are correctly set in the `.env` file.
- The scripts assume that the input and output directories exist or will be created if they do not.
- The `clean_markdown.py` script uses a system prompt to ensure that the Markdown files are cleaned according to specific rules without altering the content's meaning.
- The project is optimized for GPT-4.1 with a 1M token context window:
  - Maximum chunk size is set to 80,000 tokens for optimal balance between context quality and processing efficiency
  - Large documents are automatically split into manageable chunks when necessary
- Ensure that the `tiktoken` library is installed and available in your environment.
- The script assumes that the `markdown_output` directory exists and contains Markdown files to process.
- OpenAI library version 1.54.0+ is required for compatibility with current httpx versions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.