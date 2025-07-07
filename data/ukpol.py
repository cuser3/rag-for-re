from PyPDF2 import PdfReader
import re


# Path to the PDF file
pdf_path = "uk_uas_policy.pdf"
output_path = "ukpol.txt"

chapter_pattern = re.compile(r"CHAPTER [1-6] \|")

# Initialize the PDF reader
reader = PdfReader(pdf_path)

# Text extraction and filtering
pdf_text = ""
for page in reader.pages[10:]:
    raw_text = page.extract_text()

    # Filter lines to exclude topnotes, footnotes, dates, and page numbers
    filtered_text = "\n".join(
        line
        for line in raw_text.splitlines()
        if not (
            "OFFICIAL - Public" in line  # Topnote and footnote marker
            or "Page" in line  # Page numbering
            or line.strip().isdigit()  # Standalone numbers (page numbers)
            or chapter_pattern.search(line)  # remove CHAPTER [1-6] |
        )
    )
    pdf_text += filtered_text + "\n"

with open(output_path, "w", encoding="utf-8") as file:
    file.write(pdf_text)

print(f"Text saved to {output_path}")
