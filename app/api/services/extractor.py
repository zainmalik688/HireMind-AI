import re
from typing import Dict, List, Any, Optional

class EntityExtractor:
    """
    Extracts structured entities (Contact Info, Skills, Education, Experience)
    from cleaned resume text.
    """

    # Common technical & soft skills set for keyword matching
    SKILLS_DB = {
        # Programming & Frameworks
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "sql",
        "react", "angular", "vue", "node.js", "fastapi", "django", "flask", "express",
        "next.js", "tailwind", "bootstrap",
        
        # Data Science & AI/ML
        "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
        "pytorch", "scikit-learn", "pandas", "numpy", "opencv", "nltk", "spacy",
        "transformers", "huggingface", "llm", "rag", "q-learning", "cnn", "lstm",
        
        # Cloud, DevOps & Databases
        "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "gitlab",
        "ci/cd", "postgresql", "mysql", "mongodb", "redis", "sqlite",
        
        # Tools & Concepts
        "rest api", "graphql", "microservices", "agile", "scrum", "linux", "posix"
    }

    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        # Matches international and standard phone formats
        pattern = r'(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}'
        match = re.search(pattern, text)
        if match:
            phone = match.group(0).strip()
            # Basic sanity filter on digit count
            digits = re.sub(r'\D', '', phone)
            if 7 <= len(digits) <= 15:
                return phone
        return None

    @staticmethod
    def extract_links(text: str) -> Dict[str, Optional[str]]:
        linkedin = re.search(r'(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+', text, re.IGNORECASE)
        github = re.search(r'(https?://)?(www\.)?github\.com/[a-zA-Z0-9_-]+', text, re.IGNORECASE)
        
        return {
            "linkedin": linkedin.group(0) if linkedin else None,
            "github": github.group(0) if github else None
        }

    @classmethod
    def extract_skills(cls, text: str) -> List[str]:
        text_lower = text.lower()
        found_skills = set()
        
        for skill in cls.SKILLS_DB:
            # Word boundary check to prevent partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
                
        return sorted(list(found_skills))

    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        # Typically the candidate's name is in the first 3 lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:3]:
            # Simple heuristic: Avoid lines containing emails or numbers
            if not re.search(r'[@\d]', line) and len(line.split()) <= 4:
                # Clean header formatting characters
                clean_name = re.sub(r'[^a-zA-Z\s]', '', line).strip()
                if clean_name:
                    return clean_name
        return None

    @classmethod
    def parse_all(cls, cleaned_text: str) -> Dict[str, Any]:
        links = cls.extract_links(cleaned_text)
        return {
            "candidate_info": {
                "name": cls.extract_name(cleaned_text),
                "email": cls.extract_email(cleaned_text),
                "phone": cls.extract_phone(cleaned_text),
                "linkedin": links["linkedin"],
                "github": links["github"]
            },
            "skills": cls.extract_skills(cleaned_text),
            "total_skills_count": len(cls.extract_skills(cleaned_text))
        }