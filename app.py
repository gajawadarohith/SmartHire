import streamlit as st
import pandas as pd
import numpy as np
import os
import zipfile
import re
import spacy
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import google.generativeai as genai
import json
from dotenv import load_dotenv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import pathlib
import shutil
from difflib import SequenceMatcher
import unicodedata
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set your Gemini API key in the .env file")
    st.stop()
genai.configure(api_key=GEMINI_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Advanced Resume Screening & Ranking System",
    page_icon="📄",
    layout="wide"
)

# Create necessary directories with proper path handling
def create_directories():
    """Create necessary directories with proper error handling"""
    try:
        # Create a temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        
        # Create other directories
        for directory in ["uploaded_resumes", "processed_data", "assets"]:
            dir_path = pathlib.Path(directory)
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                except PermissionError:
                    logger.warning(f"Could not remove directory {dir_path}. It may be in use.")
                except Exception as e:
                    logger.error(f"Error removing directory {dir_path}: {e}")
            
            try:
                dir_path.mkdir(exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating directory {dir_path}: {e}")
        
        return temp_dir
    except Exception as e:
        logger.error(f"Error in directory creation: {e}")
        st.error("Error creating necessary directories. Please check permissions.")
        return None

# Create directories
TEMP_DIR = create_directories()
if not TEMP_DIR:
    st.error("Failed to initialize the application. Please check permissions and try again.")
    st.stop()

# Load spaCy model with error handling
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        st.error("Error loading language model. Please try again.")
        return None

# Load sentence transformer model with error handling
@st.cache_resource
def load_sentence_transformer():
    try:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading sentence transformer: {e}")
        return None, None

# Initialize models
nlp = load_spacy_model()
if not nlp:
    st.error("Failed to load required models. Please try again.")
    st.stop()

tokenizer, sentence_model = load_sentence_transformer()

def clean_text(text):
    """Clean and normalize text with error handling"""
    try:
        # Remove special characters and normalize unicode
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+{[^}]*}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        # Remove math mode
        text = re.sub(r'\$[^$]*\$', '', text)
        # Remove citations
        text = re.sub(r'\\cite{[^}]*}', '', text)
        # Remove references
        text = re.sub(r'\\ref{[^}]*}', '', text)
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with improved error handling"""
    text = ""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_file.read())
            temp_pdf_path = temp_pdf.name

        # Open the temporary file
        pdf_document = fitz.open(temp_pdf_path)
        
        for page_num in range(len(pdf_document)):
            try:
                page = pdf_document.load_page(page_num)
                blocks = page.get_text("blocks")
                for block in blocks:
                    block_text = clean_text(block[4])
                    if block_text:
                        text += block_text + "\n"
                text += "\n"
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue
        
        # Clean up
        pdf_document.close()
        os.unlink(temp_pdf_path)
        
        # Clean up text
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(docx_file):
    """Extract text from DOCX with improved error handling"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_docx:
            temp_docx.write(docx_file.read())
            temp_docx_path = temp_docx.name

        # Process the file
        text = docx2txt.process(temp_docx_path)
        text = clean_text(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Clean up
        os.unlink(temp_docx_path)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def calculate_string_similarity(str1, str2):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def analyze_job_description(job_description):
    """Analyze job description using Gemini API with improved error handling"""
    prompt = f"""
    Analyze the following job description and extract key requirements and criteria.
    Focus on:
    1. Required skills (technical and non-technical)
    2. Experience requirements
    3. Educational qualifications
    4. Key responsibilities
    5. Preferred qualifications
    6. Required certifications
    7. Project requirements
    
    Job Description:
    {job_description}
    
    Provide a structured analysis in JSON format with the following structure:
    {{
        "technical_skills": [],
        "soft_skills": [],
        "experience_years": 0,
        "education": [],
        "certifications": [],
        "project_requirements": []
    }}
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        
        # Clean the response text
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        # Parse JSON with error handling
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            # Return default structure if JSON parsing fails
            return {
                "technical_skills": [],
                "soft_skills": [],
                "experience_years": 0,
                "education": [],
                "certifications": [],
                "project_requirements": []
            }
    except Exception as e:
        logger.error(f"Error analyzing job description: {e}")
        return None

def process_resume(text):
    """Enhanced resume processing with improved information extraction"""
    doc = nlp(text[:100000])
    
    resume_data = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "education": [],
        "experience": [],
        "projects": [],
        "certifications": [],
        "extracurricular": [],
        "cgpa": None,
        "technical_skills": [],
        "soft_skills": [],
        "languages": [],
        "achievements": []
    }
    
    # Extract basic information with improved patterns
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\d{10}'
    
    emails = re.findall(email_pattern, text)
    if emails:
        resume_data["email"] = emails[0]
    
    phones = re.findall(phone_pattern, text)
    if phones:
        resume_data["phone"] = phones[0]
    
    # Extract CGPA with improved patterns
    cgpa_patterns = [
        r'(?:CGPA|GPA)(?:\s*[:of]\s|\s*[-=]?\s*)(\d+(?:\.\d+)?)',
        r'(?:CGPA|GPA)(?:\s*[:of]\s|\s*[-=]?\s*)(\d+(?:\.\d+)?)/\d+(?:\.\d+)?',
        r'(?:CGPA|GPA)(?:.?)(\d+\.\d+)(?:\s\/\s*\d+(?:\.\d+)?)?',
        r'(\d+(?:\.\d+)?)\s*(?:CGPA|GPA)',
        r'CGPA\s*[:=]\s*(\d+(?:\.\d+)?)',
        r'GPA\s*[:=]\s*(\d+(?:\.\d+)?)'
    ]
    
    for pattern in cgpa_patterns:
        cgpa_matches = re.findall(pattern, text, re.IGNORECASE)
        if cgpa_matches:
            try:
                cgpa_value = cgpa_matches[0].split('/')[0] if '/' in cgpa_matches[0] else cgpa_matches[0]
                resume_data["cgpa"] = float(cgpa_value)
                break
            except:
                continue
    
    # Extract sections using improved patterns
    sections = {
        "education": r'(?:EDUCATION|ACADEMIC QUALIFICATIONS|QUALIFICATIONS|EDUCATIONAL BACKGROUND|ACADEMIC)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)',
        "experience": r'(?:EXPERIENCE|WORK EXPERIENCE|EMPLOYMENT|WORK HISTORY|PROFESSIONAL EXPERIENCE)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)',
        "projects": r'(?:PROJECTS?|ACADEMIC PROJECTS?|PERSONAL PROJECTS?|RESEARCH PROJECTS?)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)',
        "certifications": r'(?:CERTIFICATIONS?|COURSES?|TRAINING|PROFESSIONAL CERTIFICATIONS?)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)',
        "extracurricular": r'(?:EXTRACURRICULAR|ACTIVITIES|VOLUNTEER|LEADERSHIP|POSITIONS OF RESPONSIBILITY)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)',
        "achievements": r'(?:ACHIEVEMENTS|AWARDS|RECOGNITIONS|HONORS|MERITS)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)'
    }
    
    for section, pattern in sections.items():
        section_match = re.search(pattern, text, re.IGNORECASE)
        if section_match:
            section_text = section_match.group().strip()
            lines = [line.strip() for line in section_text.split('\n') if line.strip()]
            if lines:
                lines.pop(0)  # Remove section header
                resume_data[section] = lines
    
    # Extract skills using improved categorization and fuzzy matching
    technical_skills = [
        "python", "java", "c\\+\\+", "javascript", "typescript", "html", "css", "react", "angular", 
        "node\\.js", "django", "flask", "express", "mongodb", "mysql", "postgresql", "redis",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "git", "devops", "ci/cd",
        "machine learning", "deep learning", "data analysis", "data science", "artificial intelligence", 
        "tensorflow", "pytorch", "keras", "nlp", "computer vision", "rust", "golang", "scala",
        "hadoop", "spark", "kafka", "tableau", "power bi", "excel", "word", "powerpoint", "sql",
        "latex", "matlab", "r", "jupyter", "pandas", "numpy", "scikit-learn", "opencv", "tensorflow",
        "pytorch", "keras", "spark", "hadoop", "kafka", "elasticsearch", "redis", "docker",
        "kubernetes", "jenkins", "git", "jira", "confluence", "agile", "scrum", "kanban"
    ]
    
    soft_skills = [
        "communication", "leadership", "teamwork", "problem solving", "critical thinking",
        "agile", "scrum", "kanban", "project management", "time management", "analytical",
        "creativity", "adaptability", "interpersonal", "negotiation", "presentation",
        "collaboration", "decision making", "conflict resolution", "emotional intelligence",
        "mentoring", "coaching", "public speaking", "writing", "research", "analysis"
    ]
    
    # Extract technical skills with fuzzy matching
    for skill in technical_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            resume_data["technical_skills"].append(skill)
        else:
            # Try fuzzy matching for similar skills
            words = text.lower().split()
            for word in words:
                if calculate_string_similarity(skill, word) > 0.8:  # 80% similarity threshold
                    resume_data["technical_skills"].append(skill)
                    break
    
    # Extract soft skills with fuzzy matching
    for skill in soft_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
            resume_data["soft_skills"].append(skill)
        else:
            # Try fuzzy matching for similar skills
            words = text.lower().split()
            for word in words:
                if calculate_string_similarity(skill, word) > 0.8:  # 80% similarity threshold
                    resume_data["soft_skills"].append(skill)
                    break
    
    # Extract languages with improved pattern
    language_pattern = r'(?:Languages?|Proficiency|Language Skills?)[\s\S]*?(?=\n[A-Z][A-Z ]+\n|\Z)'
    language_match = re.search(language_pattern, text, re.IGNORECASE)
    if language_match:
        language_text = language_match.group().strip()
        languages = re.findall(r'[A-Za-z]+(?:\s+[A-Za-z]+)*', language_text)
        resume_data["languages"] = [lang.strip() for lang in languages if lang.strip()]
    
    # Extract name using NER and fallback methods
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not resume_data["name"]:
            if text.find(ent.text) < 500:
                resume_data["name"] = ent.text
    
    if not resume_data["name"]:
        first_lines = text.strip().split('\n')[:5]
        for line in first_lines:
            line = line.strip()
            if 2 <= len(line.split()) <= 5 and re.match(r'^[A-Za-z\s.]+$', line):
                resume_data["name"] = line
                break
    
    return resume_data

def calculate_match_score(resume_data, job_requirements):
    """Calculate match score using improved scoring system"""
    score = 0
    max_score = 100
    
    # Technical Skills (35%)
    if job_requirements.get("technical_skills"):
        tech_skills_score = 0
        for req_skill in job_requirements["technical_skills"]:
            # Check for exact matches
            if req_skill.lower() in [skill.lower() for skill in resume_data["technical_skills"]]:
                tech_skills_score += 1
            # Check for partial matches
            else:
                for skill in resume_data["technical_skills"]:
                    if calculate_string_similarity(req_skill, skill) > 0.7:  # 70% similarity threshold
                        tech_skills_score += 0.5
                        break
        tech_skills_percentage = (tech_skills_score / len(job_requirements["technical_skills"])) * 35
        score += tech_skills_percentage
    
    # Soft Skills (25%)
    if job_requirements.get("soft_skills"):
        soft_skills_score = 0
        for req_skill in job_requirements["soft_skills"]:
            # Check for exact matches
            if req_skill.lower() in [skill.lower() for skill in resume_data["soft_skills"]]:
                soft_skills_score += 1
            # Check for partial matches
            else:
                for skill in resume_data["soft_skills"]:
                    if calculate_string_similarity(req_skill, skill) > 0.7:  # 70% similarity threshold
                        soft_skills_score += 0.5
                        break
        soft_skills_percentage = (soft_skills_score / len(job_requirements["soft_skills"])) * 25
        score += soft_skills_percentage
    
    # Experience (20%)
    if job_requirements.get("experience_years"):
        exp_score = min(len(resume_data["experience"]) / job_requirements["experience_years"], 1) * 20
        score += exp_score
    
    # Education (10%)
    if job_requirements.get("education"):
        edu_score = 0
        for edu in resume_data["education"]:
            if any(req_edu.lower() in edu.lower() for req_edu in job_requirements["education"]):
                edu_score += 1
        edu_percentage = (edu_score / len(job_requirements["education"])) * 10
        score += edu_percentage
    
    # Projects (10%)
    if job_requirements.get("project_requirements"):
        project_score = 0
        for project in resume_data["projects"]:
            if any(req.lower() in project.lower() for req in job_requirements["project_requirements"]):
                project_score += 1
        project_percentage = (project_score / len(job_requirements["project_requirements"])) * 10
        score += project_percentage
    
    return round(score, 2)

def save_resume_data(data, filename):
    """Save resume data to JSON with improved error handling"""
    try:
        # Create a safe filename
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', os.path.splitext(filename)[0])
        json_path = pathlib.Path("processed_data") / f"{safe_filename}.json"
        
        # Ensure the directory exists
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the data
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON for {filename}: {e}")
        return False

def process_zip_file(zip_file):
    """Process ZIP file containing resumes with improved error handling"""
    resume_data = {}
    
    try:
        with zipfile.ZipFile(zip_file) as z:
            for file_name in z.namelist():
                if file_name.endswith(('.pdf', '.docx', '.doc')) and not file_name.startswith('__MACOSX'):
                    try:
                        with z.open(file_name) as f:
                            content = BytesIO(f.read())
                            if file_name.endswith('.pdf'):
                                text = extract_text_from_pdf(content)
                            elif file_name.endswith(('.docx', '.doc')):
                                text = extract_text_from_docx(content)
                            
                            if text:
                                processed_data = process_resume(text)
                                resume_data[file_name] = processed_data
                                
                                # Save to JSON
                                save_resume_data(processed_data, file_name)
                    except Exception as e:
                        logger.error(f"Error processing file {file_name}: {e}")
                        continue
    
    except Exception as e:
        logger.error(f"Error processing ZIP file: {e}")
        st.error(f"Error processing ZIP file: {e}")
    
    return resume_data

def get_csv_download_link(df, filename="ranked_candidates.csv"):
    """Create download link for CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="btn" style="background-color:#4CAF50;color:white;padding:8px 12px;text-decoration:none;border-radius:4px;">Download Ranked Candidates CSV</a>'
    return href

def create_match_score_plot(df_results):
    """Create match score plot with improved error handling"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        df_results_sorted = df_results.sort_values(by="Match Score", ascending=False)
        
        # Create bar plot
        bars = ax.bar(range(len(df_results_sorted)), df_results_sorted["Match Score"], color='royalblue')
        
        # Set labels and title
        ax.set_xlabel("Applicants")
        ax.set_ylabel("Match Score (%)")
        ax.set_title("Match Scores of Applicants")
        
        # Set x-axis ticks and labels
        ax.set_xticks(range(len(df_results_sorted)))
        ax.set_xticklabels(df_results_sorted["Name"], rotation=45, ha="right", fontsize=8)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error creating match score plot: {e}")
        return None

def main():
    try:
        st.title("Advanced Resume Screening & Ranking System")
        
        tab1, tab2 = st.tabs(["Resume Screening", "About the System"])
        
        with tab1:
            st.header("Upload Resumes & Job Description")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Upload Resumes")
                zip_file = st.file_uploader("Upload ZIP file containing resumes", type=["zip"])
                
                if zip_file:
                    with st.spinner("Processing resumes..."):
                        resume_data = process_zip_file(zip_file)
                        if resume_data:
                            st.success(f"Successfully processed {len(resume_data)} resumes")
                        else:
                            st.error("No valid resumes found in the ZIP file")
            
            with col2:
                st.subheader("Job Description")
                job_description = st.text_area("Enter job description:", height=200)
                
                if job_description:
                    with st.spinner("Analyzing job description..."):
                        job_requirements = analyze_job_description(job_description)
                        if job_requirements:
                            st.success("Job requirements analyzed successfully")
                        else:
                            st.error("Error analyzing job description. Please try again.")
            
            if st.button("Screen & Rank Resumes") and resume_data and job_requirements:
                with st.spinner("Calculating rankings..."):
                    try:
                        results = []
                        detailed_data = {}
                        
                        for filename, data in resume_data.items():
                            match_score = calculate_match_score(data, job_requirements)
                            
                            detailed_data[filename] = {
                                "resume_data": data,
                                "match_score": match_score
                            }
                            
                            results.append({
                                "Filename": filename,
                                "Name": data["name"] or "Unknown",
                                "Email": data["email"] or "Not found",
                                "Phone": data["phone"] or "Not found",
                                "CGPA": float(data["cgpa"]) if data["cgpa"] else 0.0,
                                "Technical Skills": ", ".join(data["technical_skills"]),
                                "Soft Skills": ", ".join(data["soft_skills"]),
                                "Match Score": match_score
                            })
                        
                        # Create DataFrame and sort by match score
                        df_results = pd.DataFrame(results).sort_values(by="Match Score", ascending=False)
                        
                        # Display results
                        st.header("Ranking Results")
                        
                        # Display top candidates
                        st.subheader("Top Candidates")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Create download link for CSV
                        st.markdown(get_csv_download_link(df_results), unsafe_allow_html=True)
                        
                        # Show detailed info for top candidates
                        st.subheader("Top Candidates Details")
                        top_candidates = df_results.head(min(5, len(df_results)))
                        
                        for i, (_, row) in enumerate(top_candidates.iterrows(), 1):
                            filename = row['Filename']
                            detail = detailed_data[filename]
                            
                            with st.expander(f"#{i}: {row['Name']} - Match Score: {row['Match Score']}%"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**Contact Information:**")
                                    st.write(f"- Email: {row['Email']}")
                                    st.write(f"- Phone: {row['Phone']}")
                                    st.write("**Education:**")
                                    st.write(f"- CGPA: {row['CGPA']}")
                                    for edu in detail["resume_data"]["education"]:
                                        st.write(f"- {edu}")
                                
                                with col2:
                                    st.write("**Skills:**")
                                    st.write("Technical Skills:")
                                    for skill in detail["resume_data"]["technical_skills"]:
                                        st.write(f"- {skill}")
                                    st.write("Soft Skills:")
                                    for skill in detail["resume_data"]["soft_skills"]:
                                        st.write(f"- {skill}")
                                
                                if detail["resume_data"]["projects"]:
                                    st.write("**Projects:**")
                                    for project in detail["resume_data"]["projects"][:3]:
                                        st.write(f"- {project}")
                                
                                if detail["resume_data"]["experience"]:
                                    st.write("**Experience:**")
                                    for exp in detail["resume_data"]["experience"][:3]:
                                        st.write(f"- {exp}")
                                
                                if detail["resume_data"]["certifications"]:
                                    st.write("**Certifications:**")
                                    for cert in detail["resume_data"]["certifications"][:3]:
                                        st.write(f"- {cert}")
                                
                                if detail["resume_data"]["extracurricular"]:
                                    st.write("**Extracurricular Activities:**")
                                    for activity in detail["resume_data"]["extracurricular"][:3]:
                                        st.write(f"- {activity}")
                        
                        # Display match score distribution
                        st.subheader("Match Score Distribution")
                        fig = create_match_score_plot(df_results)
                        if fig:
                            st.pyplot(fig)
                        else:
                            st.error("Error creating match score plot")
                    
                    except Exception as e:
                        logger.error(f"Error processing results: {e}")
                        st.error("Error processing results. Please try again.")
        
        with tab2:
            st.header("About the Resume Screening System")
            
            st.write("""
            This advanced resume screening and ranking system helps employers efficiently filter through job applications
            by automatically extracting key information from resumes and ranking candidates based on job requirements.
            
            ### Key Features:
            
            1. **Intelligent Resume Processing**:
               - Extracts information from PDF and DOCX formats
               - Preserves document structure
               - Stores processed data in JSON format
            
            2. **Advanced Job Description Analysis**:
               - Uses Gemini API for intelligent analysis
               - Extracts key requirements and criteria
               - Identifies technical and soft skills
            
            3. **Comprehensive Matching System**:
               - Technical skills matching (35%)
               - Soft skills matching (25%)
               - Experience evaluation (20%)
               - Education matching (10%)
               - Project relevance (10%)
            
            4. **Detailed Candidate Profiles**:
               - Contact information
               - Educational background
               - Technical and soft skills
               - Project experience
               - Work experience
               - Certifications
               - Extracurricular activities
            
            5. **Visual Analytics**:
               - Match score distribution
               - Detailed candidate comparisons
               - Exportable results
            
            6. **Bulk Processing**:
               - Supports ZIP file uploads
               - Processes multiple resumes simultaneously
               - Generates comprehensive reports
            """)
            
            st.info("""
            *Note*: This system is designed to assist in the initial screening process. 
            It is recommended to review the top candidates manually for a more comprehensive evaluation.
            """)
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error("An unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main()
