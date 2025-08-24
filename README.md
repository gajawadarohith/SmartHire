# Advanced Resume Screening & Ranking System

An intelligent system that automatically screens and ranks resumes based on job requirements using advanced NLP techniques and AI.

## Features

- **Intelligent Resume Processing**
  - Extracts information from PDF and DOCX formats
  - Preserves document structure
  - Stores processed data in JSON format

- **Advanced Job Description Analysis**
  - Uses Gemini API for intelligent analysis
  - Extracts key requirements and criteria
  - Identifies technical and soft skills

- **Comprehensive Matching System**
  - Technical skills matching (30%)
  - Soft skills matching (20%)
  - Experience evaluation (20%)
  - Education matching (15%)
  - Project relevance (15%)

- **Detailed Candidate Profiles**
  - Contact information
  - Educational background
  - Technical and soft skills
  - Project experience
  - Work experience
  - Certifications
  - Extracurricular activities

- **Bulk Processing**
  - Supports ZIP file uploads
  - Processes multiple resumes simultaneously
  - Generates comprehensive reports

## Setup

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd resume-screening
    ```

2. Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    # On Unix/macOS:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root and add your Gemini API key:
    ```
    API_KEY=your_gemini_api_key_here
    ```

## Usage

1. Start the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Upload a ZIP file containing resumes (PDF or DOCX format)

4. Enter the job description in the text area

5. Click "Screen & Rank Resumes" to process the resumes

6. View the results, including:
   - Top candidates
   - Detailed candidate profiles
   - Match score distribution
   - Downloadable CSV report

## Directory Structure

```
resume-screening/
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── Dataset/               # Sample resumes
```

## License

This project is licensed under the MIT License - see the LICENSE
