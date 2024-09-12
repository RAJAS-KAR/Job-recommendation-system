import pandas as pd
import fitz
from sentence_transformers import SentenceTransformer, util
import re
import spacy

model = SentenceTransformer('paraphrase-Mini LM-L6-v2')

nlp = spacy.load("en_core_web_sm")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", clean_up_tokenization_spaces= True)

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def parse_resume(text):
    skills = []
    job_title = "UnKnown"
    qualifications = "UnKnown"
    experience = 0

    skill_patterns = re.compile(
        r'\b(Python|Machine Learning|SQL|Java|JavaScript|HTML|CSS|Node\.js|React|AWS|'
        r'UI/UX|Docker|Kubernetes|C\+\+|C#|Ruby|PHP|TypeScript|Go|Swift|Objective-C|'
        r'Angular|Vue\.js|Django|Flask|TensorFlow|PyTorch|Spark|Hadoop|Git|CI/CD|'
        r'PostgreSQL|MySQL|MongoDB|NoSQL|Terraform|Ansible|Jenkins|Azure|GCP|'
        r'Agile|Scrum|DevOps|JIRA|Tableau|Power BI|R|SAS|SPSS|Scala)\b', re.IGNORECASE)

    job_title_patterns = re.compile(
        r'\b(Data Scientist|Software Engineer|Frontend Developer|Backend Developer|'
        r'Project Manager|Data Analyst|UX Designer|DevOps Engineer|System Architect|'
        r'Full Stack Developer|Cloud Engineer|AI Engineer|ML Engineer|Data Engineer|'
        r'Product Manager|QA Engineer|Security Engineer|Network Engineer|UI Designer|'
        r'Mobile Developer|Game Developer|Database Administrator|IT Support Specialist|'
        r'Tech Lead|Scrum Master|Digital Marketing Executive)\b', re.IGNORECASE)

    qualification_patterns = re.compile(
        r'\b(PhD|MSc|BSc|MBA|BA|BTech|BE|MTech|BCA|MCA|Diploma|'
        r'Certification in Data Science|Certified Scrum Master|'
        r'AWS Certified Solutions Architect|Microsoft Certified: Azure Administrator Associate|'
        r'Google Cloud Certified|Cisco Certified Network Associate|'
        r'Certified Information Systems Security Professional \(CISSP\)|Certified Ethical Hacker \(CEH\))\b',
        re.IGNORECASE)

    doc = nlp(text)

    for ent in doc.ents:
        if skill_patterns.search(ent.text):
            skills.append(ent.text)
        if ent.label_ == "JOB_TITLE" or job_title_patterns.search(ent.text):
            job_title = ent.text
        if qualification_patterns.search(ent.text):
            qualifications = ent.text
    if job_title == "UnKnown":
        job_title_matches = job_title_patterns.findall(text)
        if job_title_matches:
            job_title = job_title_matches[0]

    experience_match = re.search(r'(\d+)\s+years', text, re.IGNORECASE)
    if experience_match:
        experience = int(experience_match.group(1))

    return {
        'skills': ', '.join(skills) if skills else 'UnKnown',
        'job_title': job_title,
        'qualifications': qualifications,
        'experience': experience
    }

def resume_to_dataframe(parsed_info):
    return pd.DataFrame([parsed_info])

job_postings = pd.DataFrame([
    {'job_id': 1, 'company': 'TechCorp', 'job_title': 'Data Scientist', 'required_skills': 'Python, Machine Learning, SQL', 'min_experience': 3, 'max_experience': 7, 'qualifications': 'PhD in Computer Science'},
    {'job_id': 2, 'company': 'FinTech Inc.', 'job_title': 'Java Developer', 'required_skills': 'Java, Spring, Hibernate', 'min_experience': 2, 'max_experience': 5, 'qualifications': 'BSc in Computer Science'},
    {'job_id': 3, 'company': 'WebSolutions', 'job_title': 'Frontend Developer', 'required_skills': 'HTML, CSS, JavaScript', 'min_experience': 2, 'max_experience': 6, 'qualifications': 'BA in Graphic Design'},
    {'job_id': 4, 'company': 'DevOps LLC', 'job_title': 'Backend Developer', 'required_skills': 'Node.js, Express, MongoDB', 'min_experience': 1, 'max_experience': 4, 'qualifications': 'MSc in Computer Science'},
    {'job_id': 5, 'company': 'Management Solutions', 'job_title': 'Project Manager', 'required_skills': 'Project Management, Agile, Scrum', 'min_experience': 5, 'max_experience': 10, 'qualifications': 'MBA'},
    {'job_id': 6, 'company': 'AnalyticsPro', 'job_title': 'Data Analyst', 'required_skills': 'Excel, SQL, Tableau', 'min_experience': 2, 'max_experience': 4, 'qualifications': 'BSc in Mathematics'},
    {'job_id': 7, 'company': 'DesignHub', 'job_title': 'UX Designer', 'required_skills': 'Sketch, Figma, UX Research', 'min_experience': 3, 'max_experience': 6, 'qualifications': 'BA in Graphic Design'},
    {'job_id': 8, 'company': 'CloudNet', 'job_title': 'DevOps Engineer', 'required_skills': 'AWS, Docker, Jenkins, Kubernetes', 'min_experience': 2, 'max_experience': 6, 'qualifications': 'BSc in Information Technology'},
    {'job_id': 9, 'company': 'HealthTech', 'job_title': 'Data Engineer', 'required_skills': 'SQL, Python, ETL, Big Data', 'min_experience': 3, 'max_experience': 6, 'qualifications': 'MSc in Data Science'},
    {'job_id': 10, 'company': 'CyberSecure', 'job_title': 'Cybersecurity Analyst', 'required_skills': 'Network Security, Ethical Hacking, Python', 'min_experience': 2, 'max_experience': 5, 'qualifications': 'BSc in Cybersecurity'},
    {'job_id': 11, 'company': 'FinancePro', 'job_title': 'Financial Analyst', 'required_skills': 'Excel, Financial Modeling, SQL', 'min_experience': 1, 'max_experience': 3, 'qualifications': 'BCom in Finance'},
    {'job_id': 12, 'company': 'MedSoft', 'job_title': 'Software Engineer', 'required_skills': 'Java, C++, Healthcare Systems', 'min_experience': 3, 'max_experience': 7, 'qualifications': 'BSc in Software Engineering'},
    {'job_id': 13, 'company': 'EduTech', 'job_title': 'Instructional Designer', 'required_skills': 'E-learning, Curriculum Design, LMS', 'min_experience': 2, 'max_experience': 5, 'qualifications': 'MA in Education Technology'},
    {'job_id': 14, 'company': 'GreenEnergy', 'job_title': 'Environmental Scientist', 'required_skills': 'Environmental Science, Data Analysis, GIS', 'min_experience': 2, 'max_experience': 5, 'qualifications': 'MSc in Environmental Science'},
    {'job_id': 15, 'company': 'Adstech', 'job_title': 'Digital Marketing Executive', 'required_skills': 'Google Analytics, SEMrush, Mailchimp, WordPress, Salesforce, Google Ads, Adobe Creative Cloud,HubSpot', 'min_experience': 3, 'max_experience': 6, 'qualifications': 'Bachelor of Science Marketing'}
])

def recommend_jobs_based_on_resume(pdf_path, job_postings):
    resume_text = extract_text_from_pdf(pdf_path)
    parsed_info = parse_resume(resume_text)
    user_profile = resume_to_dataframe(parsed_info)

    user_embedding = model.encode(user_profile['skills'].iloc[0])
    job_embeddings = model.encode(job_postings['required_skills'].tolist())
    cosine_similarities = util.pytorch_cos_sim(user_embedding, job_embeddings)

    def get_overall_similarity(user_profile, cosine_similarities, job_postings, top_n=5):
        similarity_scores = []

        for job_index, job in job_postings.iterrows():
            skill_similarity = cosine_similarities[0][job_index].item() * 100
            experience_eligibility = job['min_experience'] <= user_profile['experience'][0] <= job['max_experience']
            experience_similarity = 100 if experience_eligibility else 0
            qualification_similarity = 100 if user_profile['qualifications'][0] == job['qualifications'] else 0
            overall_similarity = (skill_similarity * 0.5) + (experience_similarity * 0.3) + (qualification_similarity * 0.2)
            similarity_scores.append((job_index, overall_similarity))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_job_indices = [i[0] for i in similarity_scores[:top_n]]
        top_job_similarities = [i[1] for i in similarity_scores[:top_n]]

        recommendations = job_postings.iloc[top_job_indices].copy()
        recommendations['similarity'] = top_job_similarities

        return recommendations, user_profile

    recommendations, user_profile = get_overall_similarity(user_profile, cosine_similarities, job_postings)
    return recommendations, user_profile

def print_recommendations(recommendations):
    print("Recommended jobs based on resume:")
    for _, row in recommendations.iterrows():
        print(f"  Company: {row['company']}, Position: {row['job_title']}, Similarity: {row['similarity']:.2f}%")
pdf_path = r"C:\Users\R RAJA SEKAR\Downloads\ui-ux-designer-resume-example.pdf"
recommended_jobs, user_profile = recommend_jobs_based_on_resume(pdf_path, job_postings)
print("\nFinal User Profile DataFrame:\n", user_profile)
print_recommendations(recommended_jobs)