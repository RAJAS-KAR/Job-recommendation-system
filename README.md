
# Resume Parsing and Job Recommendation System

This repository contains a Python script that parses resumes in PDF format, extracts relevant information like skills, job titles, qualifications, and experience, and then recommends matching job postings using sentence embeddings.

## Features

- **PDF Resume Parsing**: Extracts key information such as skills, job titles, qualifications, and experience.
- **Job Recommendations**: Recommends relevant job postings based on the parsed resume information.
- **Embeddings for Matching**: Uses the SentenceTransformer model to match the parsed resume with job postings based on skills, qualifications, and experience.

## How It Works

1. **Extract Text from Resume**: The script uses the PyMuPDF library (`fitz`) to extract text from a resume PDF.
2. **Segment and Parse Resume**: The script segments the resume by key sections (Skills, Experience, Education) and uses regular expressions and spaCy for parsing.
3. **Fuzzy Matching**: Qualifications are matched using fuzzy logic to account for variations in how degrees and certifications are presented.
4. **Job Matching**: Sentence embeddings are used to compute similarity between the resume and job postings, considering skills, experience, and qualifications.
5. **Results**: The top job matches are printed based on the overall similarity score.

### Example

**User Profile Extracted from Resume:**
```text
skills: "UI/UX Designer, HTML/CSS, Fixed JavaScript, Jira, UserTesting, Adobe XD"
job_title: "UI/UX Designer"
qualifications: "Bachelor of Design"
experience: 5 years
```

**Recommended Jobs Based on Resume:**
```text
Company: WebSolutions, Position: Frontend Developer, Similarity: 21.56%
Company: DesignHub, Position: UX Designer, Similarity: 11.95%
Company: Adstech, Position: Digital Marketing Executive, Similarity: 10.34%
Company: DevOps LLC, Position: Backend Developer, Similarity: 8.98%
Company: Management Solutions, Position: Project Manager, Similarity: 4.51%
```
## Dependencies

- Python 3.x
- Pandas
- SentenceTransformers
- PyMuPDF (fitz)
- spaCy
- FuzzyWuzzy
- Transformers

## Usage

1. Place the resume PDF in the project folder.
2. Update the `pdf_path` variable in the script to point to your resume file.
3. Run the script to extract the resume information and receive job recommendations.
