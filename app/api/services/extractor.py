import re
from typing import Dict, List, Any, Optional


class EntityExtractor:
    """
    Extracts structured entities (Contact Info, Skills, Education, Experience,
    Projects, Certifications) from cleaned resume text.
    """

    SKILLS_DB = {
        # Programming & Frameworks
        "python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "sql",
        "react", "angular", "vue", "node.js", "fastapi", "django", "flask", "express",
        "next.js", "tailwind", "bootstrap",
        
        # Data Science & AI/ML
        "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
        "pytorch", "scikit-learn", "pandas", "numpy", "opencv", "nltk", "spacy",
        "transformers", "huggingface", "llm", "rag", "q-learning", "cnn", "cnns", "lstm",
        "peft", "lora",
        
        # Cloud, DevOps & Databases
        "docker", "kubernetes", "aws", "azure", "gcp", "git", "github", "gitlab",
        "ci/cd", "postgresql", "mysql", "mongodb", "redis", "sqlite",
        
        # Tools & Concepts
        "rest api", "graphql", "microservices", "agile", "scrum", "linux", "ubuntu", "posix"
    }

    SECTION_HEADERS = {
        "experience": [
            "EXPERIENCE", "WORK EXPERIENCE", "EMPLOYMENT HISTORY", "WORK HISTORY", "PROFESSIONAL EXPERIENCE"
        ],
        "education": [
            "EDUCATION", "ACADEMIC BACKGROUND", "ACADEMIC QUALIFICATIONS", "EDUCATION AND TRAINING"
        ],
        "skills": [
            "TECHNICAL SKILLS", "SKILLS", "CORE COMPETENCIES", "SKILLS & INTERESTS", "COMPETENCIES"
        ],
        "projects": [
            "PROJECTS", "PERSONAL PROJECTS", "ACADEMIC PROJECTS", "KEY PROJECTS"
        ],
        "certifications": [
            "CERTIFICATIONS", "LICENSES & CERTIFICATIONS", "CERTIFICATES", "COURSES & CERTIFICATIONS"
        ],
        "community": [
            "COMMUNITY & VOLUNTEERING", "VOLUNTEERING", "COMMUNITY ENGAGEMENT", "EXTRA-CURRICULAR"
        ],
    }

    @staticmethod
    def extract_email(text: str) -> Optional[str]:
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    @staticmethod
    def extract_phone(text: str) -> Optional[str]:
        pattern = r'(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}'
        match = re.search(pattern, text)
        if match:
            phone = match.group(0).strip()
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
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
                
        return sorted(list(found_skills))

    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:3]:
            if not re.search(r'[@\d]', line) and len(line.split()) <= 4:
                clean_name = re.sub(r'[^a-zA-Z\s.]', '', line).strip()
                if clean_name:
                    return clean_name
        return None

    @classmethod
    def _extract_section_blocks(cls, text: str) -> Dict[str, str]:
        lines = text.split("\n")
        sections: Dict[str, List[str]] = {}
        current_section = "HEADER"
        sections[current_section] = []

        header_map = {}
        for sec, headers in cls.SECTION_HEADERS.items():
            for h in headers:
                header_map[h] = sec

        for line in lines:
            clean_line = line.strip().upper()
            matched_section = None
            for h_text, sec_key in header_map.items():
                if clean_line == h_text or clean_line.startswith(h_text + " ") or clean_line.endswith(" " + h_text):
                    matched_section = sec_key
                    break

            if matched_section:
                current_section = matched_section
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items()}

    @classmethod
    def parse_work_experience(cls, exp_text: str) -> List[Dict[str, Any]]:
        if not exp_text:
            return []

        entries = []
        lines = [line.strip() for line in exp_text.split("\n") if line.strip()]
        current_entry: Optional[Dict[str, Any]] = None
        
        date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\s*[\s–\-]+\s*(?:Present|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}|\d{4})|\b\d{4}\s*[\s–\-]+\s*(?:Present|\d{4})|\b\d{4}\b)'

        for line in lines:
            normalized_line = re.sub(r'\s+', ' ', line).strip()
            is_bullet_symbol = bool(re.match(r'^[•\-\*]', normalized_line))
            clean_line = re.sub(r'^[•\-\*\s]+', '', normalized_line).strip()
            dates_found = re.findall(date_pattern, normalized_line, re.IGNORECASE)

            if dates_found and len(normalized_line) < 50 and not is_bullet_symbol:
                full_date_str = dates_found[0]
                if current_entry:
                    if not current_entry["dates"]:
                        current_entry["dates"] = full_date_str
                    else:
                        entries.append(current_entry)
                        current_entry = {"title": "", "company": "", "dates": full_date_str, "bullets": []}
                else:
                    current_entry = {"title": "Role / Position", "company": "", "dates": full_date_str, "bullets": []}

            elif not dates_found and len(clean_line) < 80 and not is_bullet_symbol and not clean_line.endswith('.'):
                if current_entry and current_entry["bullets"]:
                    entries.append(current_entry)
                    current_entry = {"title": clean_line, "company": "", "dates": "", "bullets": []}
                elif current_entry and not current_entry["title"]:
                    current_entry["title"] = clean_line
                elif current_entry and not current_entry["company"]:
                    current_entry["company"] = clean_line
                else:
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {"title": clean_line, "company": "", "dates": "", "bullets": []}

            elif current_entry:
                if not current_entry["company"] and not is_bullet_symbol and len(clean_line) < 80:
                    current_entry["company"] = clean_line
                elif current_entry["bullets"] and not current_entry["bullets"][-1].rstrip().endswith(('.', ':', ';')) and not is_bullet_symbol:
                    current_entry["bullets"][-1] = f"{current_entry['bullets'][-1]} {clean_line}"
                else:
                    current_entry["bullets"].append(clean_line)

        if current_entry:
            entries.append(current_entry)

        return entries

    @classmethod
    def parse_education(cls, edu_text: str) -> List[Dict[str, Any]]:
        if not edu_text:
            return []

        edu_entries = []
        lines = [line.strip() for line in edu_text.split("\n") if line.strip()]
        degree_keywords = ["BS", "MS", "B.Sc", "M.Sc", "Bachelor", "Master", "Ph.D", "Diploma", "Associate"]
        cgpa_pattern = r'(?:CGPA|GPA)[\s:]*([0-9]\.[0-9]{1,2}(?:\s*/\s*[0-9]\.[0-9])?)'
        inst_keywords = ["University", "College", "Institute", "School", "Academy"]
        
        current_edu: Optional[Dict[str, Any]] = None
        in_coursework_block = False

        for line in lines:
            line_lower = line.lower()
            
            # Detect starting lines for subject/coursework listings
            if any(k in line_lower for k in ["core subjects", "subjects:", "courses:", "relevant coursework"]):
                in_coursework_block = True
                continue

            has_degree = any(re.search(r'\b' + re.escape(deg) + r'\b', line, re.IGNORECASE) for deg in degree_keywords)
            cgpa_match = re.search(cgpa_pattern, line, re.IGNORECASE)

            if has_degree:
                if current_edu:
                    edu_entries.append(current_edu)
                
                in_coursework_block = False

                found_inst = ""
                for kw in inst_keywords:
                    if kw.lower() in line_lower:
                        clean_line = re.sub(cgpa_pattern, '', line, flags=re.IGNORECASE).strip()
                        found_inst = re.sub(r'\s+', ' ', clean_line)
                        break

                current_edu = {
                    "degree": line,
                    "institution": found_inst,
                    "cgpa": cgpa_match.group(1) if cgpa_match else "",
                    "dates": ""
                }
            elif current_edu:
                if cgpa_match and not current_edu["cgpa"]:
                    current_edu["cgpa"] = cgpa_match.group(1)
                elif re.search(r'\b(20\d{2}|19\d{2})\b', line) and not current_edu["dates"]:
                    current_edu["dates"] = re.sub(r'\s+', ' ', line).strip()
                elif any(kw.lower() in line_lower for kw in inst_keywords):
                    clean_inst = re.sub(cgpa_pattern, '', line, flags=re.IGNORECASE).strip()
                    current_edu["institution"] = re.sub(r'\s+', ' ', clean_inst)
                elif not current_edu["institution"] and not in_coursework_block and not re.search(r'\b(20\d{2}|19\d{2})\b', line):
                    current_edu["institution"] = re.sub(r'\s+', ' ', line)

        if current_edu:
            edu_entries.append(current_edu)

        return edu_entries

    @classmethod
    def parse_projects(cls, proj_text: str) -> List[Dict[str, Any]]:
        if not proj_text:
            return []

        projects = []
        lines = [line.strip() for line in proj_text.split("\n") if line.strip()]
        current_proj: Optional[Dict[str, Any]] = None

        for line in lines:
            clean_line = re.sub(r'^[•\-\*\s]+', '', line).strip()
            if not clean_line:
                continue

            parts = re.split(r'\s+[—–\-]\s+|\s{3,}', clean_line, maxsplit=1)

            if len(parts) == 2:
                if current_proj:
                    projects.append(current_proj)
                current_proj = {
                    "name": parts[0].strip(),
                    "description": parts[1].strip()
                }
            elif current_proj:
                current_proj["description"] = f"{current_proj['description']} {clean_line}".strip()
            else:
                current_proj = {
                    "name": clean_line,
                    "description": ""
                }

        if current_proj:
            projects.append(current_proj)

        return projects

    @classmethod
    def parse_certifications(cls, cert_text: str) -> List[str]:
        if not cert_text:
            return []

        certifications = []
        lines = [line.strip() for line in cert_text.split("\n") if line.strip()]

        for line in lines:
            clean_line = re.sub(r'^[•\-\*\s]+', '', line).strip()
            if clean_line:
                certifications.append(clean_line)

        return certifications

    @classmethod
    def parse_all(cls, cleaned_text: str) -> Dict[str, Any]:
        links = cls.extract_links(cleaned_text)
        sections = cls._extract_section_blocks(cleaned_text)
        skills_found = cls.extract_skills(cleaned_text)

        return {
            "candidate_info": {
                "name": cls.extract_name(cleaned_text),
                "email": cls.extract_email(cleaned_text),
                "phone": cls.extract_phone(cleaned_text),
                "linkedin": links["linkedin"],
                "github": links["github"]
            },
            "skills": skills_found,
            "total_skills_count": len(skills_found),
            "work_experience": cls.parse_work_experience(sections.get("experience", "")),
            "education": cls.parse_education(sections.get("education", "")),
            "projects": cls.parse_projects(sections.get("projects", "")),
            "certifications": cls.parse_certifications(sections.get("certifications", "")),
        }