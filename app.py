import streamlit as st
import PyPDF2
from transformers import pipeline
import re
import time

# --- Helper Functions (keep these the same) ---
@st.cache_resource
def load_model():
    """Loads a zero-shot classification model from Hugging Face."""
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return classifier

def get_resume_score(resume_text, job_description):
    """
    Analyzes the resume against a job description using a zero-shot classifier.
    """
    classifier = load_model()
    candidate_labels = [
        "Strong technical skills",
        "Experience with project management",
        "Clear and concise summary",
        "Measurable achievements included",
        "Well-structured and easy to read",
        "Relevant keywords for the job",
        "Excellent communication skills"
    ]
    
    prompt = f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}"
    
    result = classifier(prompt, candidate_labels, multi_label=True)

    total_score = sum(result['scores'])
    average_score = total_score / len(result['scores']) * 100

    return average_score, result

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text()
    return text

# --- New Function to Extract Specific Data ---
def extract_data_from_text(text):
    """Extracts key information using regular expressions and basic heuristics."""
    data = {
        'name': 'Not found',
        'position': 'Not found',
        'email': 'Not found',
        'experience': 'Not found',
        'skills': 'Not found'
    }

    # Regex patterns
    name_pattern = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+')
    email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+')
    experience_pattern = re.compile(r'(\d+)\s+year[s]?\s+experience')
    
    # Try to find matches
    name_match = name_pattern.search(text)
    if name_match:
        data['name'] = name_match.group()
        
    email_match = email_pattern.search(text)
    if email_match:
        data['email'] = email_match.group()
        
    experience_match = experience_pattern.search(text.lower())
    if experience_match:
        data['experience'] = experience_match.group(1) + " years"

    # For position and skills, it's more complex, so we'll use heuristics
    # This is a simple placeholder and may not work for all resumes
    lines = text.split('\n')
    for line in lines:
        if ('experience' not in line.lower() and 'education' not in line.lower() and 'skills' not in line.lower() and 'contact' not in line.lower()) and len(line) < 50:
             if 'position' in data and data['position'] == 'Not found':
                 data['position'] = line.strip()
                 break
    
    if 'Skills' in text:
        data['skills'] = "Explicit Skills section found."
    else:
        data['skills'] = "No explicit skills section is present; consider adding one."
    
    return data

# --- New Function for Rule-Based Feedback ---
def get_detailed_feedback(text):
    """Generates rule-based feedback based on common resume mistakes."""
    feedback = {
        'grammatical_errors': [],
        'professional_tone': [],
        'unnecessary_information': [],
        'experience_relevance': [],
        'skills_relevance': []
    }

    # Example Grammar Rule
    if "years experience" in text.lower() and "years of experience" not in text.lower():
        feedback['grammatical_errors'].append("The phrase 'years experience' should be 'years of experience' for correct grammar.")
    
    # Example Professional Tone Rule
    if "Determined work placement" in text:
        feedback['professional_tone'].append("The phrase 'Determined work placement' could be stronger as 'Determined and assigned work placements' using stronger action verbs.")
    
    # Example Unnecessary Information Rule
    if "gpa" in text.lower() and len(re.findall(r'gpa', text.lower())) > 1:
        feedback['unnecessary_information'].append("The detailed GPA breakdown may be excessive unless applying for academic roles; consider summarizing.")

    # Example Experience Relevance Rule
    if ("childcare" in text.lower() and "adult care" in text.lower()) or ("teacher" in text.lower() and "caregiver" in text.lower()):
        feedback['experience_relevance'].append("The resume mixes different types of experience; consider tailoring or separating sections more clearly.")
    
    # Example Skills Relevance Rule
    if 'skills' not in text.lower():
        feedback['skills_relevance'].append("No explicit skills section is present; important skills should be clearly listed to improve ATS parsing.")
    
    return feedback

# --- Page Configuration and Visuals ---
st.set_page_config(
    page_title="Resume Reviewer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# --- UI Layout ---
st.markdown('<div class="title-container"><h1>Final Project Demo: Resume Reviewer</h1><h3>An AI-Powered Resume Analysis Tool</h3></div>', unsafe_allow_html=True)

st.sidebar.header("Resume Analysis Tool")
uploaded_file = st.sidebar.file_uploader("1. Upload your Resume (PDF)", type=["pdf"])
job_description = st.sidebar.text_area("2. Enter the Job Description", height=250)
analyze_button = st.sidebar.button("Analyze Resume")

# Main content area
if analyze_button:
    if uploaded_file is not None and job_description:
        with st.spinner("Analyzing your resume..."):
            time.sleep(3) # Simulate processing time
            resume_text = extract_text_from_pdf(uploaded_file)
            score, _ = get_resume_score(resume_text, job_description)
            extracted_data = extract_data_from_text(resume_text)
            feedback = get_detailed_feedback(resume_text)

            # --- Display Results ---
            st.markdown('<div class="stCard" style="animation: fadeIn 1s ease-in-out;">', unsafe_allow_html=True)
            st.success("✅ Analysis Complete!")

            # Score section
            st.header("Result")
            st.markdown(f"""
            <div style="text-align: center; margin-top: 2rem;">
                <h2 style="color: #004d99; font-size: 2rem;">Score</h2>
                <h1 style="font-size: 4rem; font-weight: 700; color: #28a745;">{int(score)}</h1>
            </div>
            """, unsafe_allow_html=True)

            # Extracted Data section
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Extracted Data")
            st.markdown(f"""
            <ul>
                <li><b>Name:</b> {extracted_data['name']}</li>
                <li><b>Position:</b> {extracted_data['position']}</li>
                <li><b>Email:</b> {extracted_data['email']}</li>
                <li><b>Experience:</b> {extracted_data['experience']}</li>
                <li><b>Skills:</b> {extracted_data['skills']}</li>
            </ul>
            <p><i><b>Note:</b> If the extracted data is incorrect, please update your resume layout. This indicates that the ATS may not be reading your resume correctly.</i></p>
            """, unsafe_allow_html=True)

            # Detailed Feedback section
            st.markdown("<hr>", unsafe_allow_html=True)
            st.header("Detailed Feedback")
            
            def display_feedback_list(title, items):
                if items:
                    st.markdown(f"**{title}**")
                    for item in items:
                        st.markdown(f"* {item}")

            display_feedback_list("Grammatical errors", feedback['grammatical_errors'])
            display_feedback_list("Professional tone", feedback['professional_tone'])
            display_feedback_list("Unnecessary information", feedback['unnecessary_information'])
            display_feedback_list("Experience relevance", feedback['experience_relevance'])
            display_feedback_list("Skills relevance", feedback['skills_relevance'])
            
            st.markdown('</div>', unsafe_allow_html=True)

    elif uploaded_file is None:
        st.error("Please upload a resume file.")
    elif not job_description:
        st.error("Please provide a job description.")

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.info("Developed BY SUNNY JADHAO 😎 using Streamlit and Hugging Face.")