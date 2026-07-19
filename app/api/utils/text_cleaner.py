import re

class TextCleaner:
    @staticmethod
    def clean_resume_text(text: str) -> str:
        if not text:
            return ""
        
        # Normalize Unicode variations and soft hyphens
        text = text.replace("\xad", "").replace("\u2013", "-").replace("\u2014", "-")
        
        # Remove excessive whitespace/newlines while preserving structural paragraph breaks
        text = re.sub(re.compile(r'\n{3,}'), '\n\n', text)
        text = re.sub(re.compile(r'[ \t]+'), ' ', text)
        
        # Strip trailing/leading spaces on each individual line
        cleaned_lines = [line.strip() for line in text.splitlines()]
        
        return "\n".join(cleaned_lines).strip()