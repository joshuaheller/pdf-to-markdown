from pathlib import Path

cleaned_dir = Path("data/markdown_cleaned")
output_dir = Path("markdown_output")

cleaned_names = {p.name for p in cleaned_dir.glob("*.md")}
for md_file in output_dir.glob("*.md"):
    if md_file.name in cleaned_names:
        try:
            md_file.unlink()
            print(f"Deleted: {md_file}")
        except Exception as e:
            print(f"Failed to delete {md_file}: {e}")