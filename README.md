# Job-recommendation-system

provides a solution to extract information from a resume PDF, parse it to identify key details such as skills, job title, qualifications, and experience, and then match the resume against job postings using sentence embeddings for skill matching. Additionally, it calculates an overall similarity score based on skills, experience, and qualifications and returns job recommendations.

Here's a quick overview:

PDF Text Extraction: Extracts text from a PDF resume.
Resume Parsing: Identifies key components like skills, job titles, qualifications, and experience using regular expressions and spaCy's NLP model.
Job Matching: Uses SentenceTransformer to encode the parsed resume's skills and compares them with job postings' required skills. The overall similarity is calculated based on skill match, experience, and qualifications.
Recommendation Generation: The top matching jobs are displayed with their similarity scores.
