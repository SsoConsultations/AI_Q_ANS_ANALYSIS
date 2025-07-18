import streamlit as st
import pandas as pd
import pdfplumber
import os
import tempfile
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer, util
from docx import Document
from openpyxl import Workbook
import openai
import re

# -------------------------
# Utility function for image encoding
# -------------------------
def get_image_as_base64(path):
    """
    Reads an image file and returns its Base64 encoded string.
    """
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Logo file '{path}' not found. Please ensure it's in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error reading logo file: {e}")
        return None

# -------------------------
# Custom CSS for logo and copyright
# -------------------------
def inject_custom_css(logo_path="SsoLogo.jpg"):
    """
    Injects custom CSS to display a logo at the top right corner
    and a copyright notice at the bottom right.
    """
    logo_base64 = get_image_as_base64(logo_path)
    
    css = """
    <style>
    /* General styling for the main Streamlit container */
    .main .block-container {
        padding-top: 1rem; /* Adjust top padding if needed due to logo position */
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }

    /* Logo positioning (Top Right) */
    .logo-container {
        position: fixed; /* Fixed position relative to the viewport */
        top: 10px; /* Distance from the top */
        right: 20px; /* Distance from the right */
        width: 80px; /* Set logo width */
        height: auto; /* Maintain aspect ratio */
        z-index: 9999; /* Ensure it's on top of almost everything */
        background-color: transparent; /* Ensure no background interferes */
    }
    .logo-container img {
        width: 100%;
        height: 100%;
        object-fit: contain; /* Ensures the entire image fits within the container */
    }

    /* Copyright positioning (Bottom Right) */
    .copyright-footer {
        position: fixed; /* Fixed position relative to the viewport */
        bottom: 10px; /* Distance from the bottom */
        right: 20px; /* Distance from the right */
        font-size: 0.8em;
        color: #888; /* Slightly muted color */
        z-index: 9998; /* Below the logo but still on top */
    }
    </style>
    """

    if logo_base64:
        css += f"""
        <div class="logo-container">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Company Logo">
        </div>
        """
    
    css += """
    <div class="copyright-footer">
        © Copyright 2025 SSO Consultants
    </div>
    """
    
    st.markdown(css, unsafe_allow_html=True)


# -------------------------
# Initialize NLP Model
# -------------------------
@st.cache_resource
def load_model():
    """
    Loads and caches the SentenceTransformer model 'all-MiniLM-L6-v2'.
    This model is used for calculating semantic similarity between text snippets.
    """
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------
# Extract text from PDF
# -------------------------
def extract_text_from_pdf(file):
    """
    Extracts all text from a given PDF file object.

    Args:
        file: A file-like object (e.g., from st.file_uploader) representing a PDF.

    Returns:
        A string containing all extracted text, with pages separated by newlines.
    """
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""
    return text

# -------------------------
# Extract text from DOCX
# -------------------------
def extract_text_from_docx(file):
    """
    Extracts all text from a given DOCX file object.

    Args:
        file: A file-like object (e.g., from st.file_uploader) representing a DOCX.

    Returns:
        A string containing all extracted text, with paragraphs separated by newlines.
    """
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# -------------------------
# Extract questions from rubric
# -------------------------
def parse_rubric_questions(text):
    """
    Parses the extracted text from the rubric file to identify questions and their max marks.
    It expects questions to be formatted like "Q#: Question text (Max Marks: #)".

    Args:
        text (str): The full text extracted from the rubric file.

    Returns:
        A list of tuples, where each tuple contains (question_number_str, question_text, max_marks_float).
    """
    question_pattern = re.compile(r"Q(\d+)\s*[^A-Za-z0-9\n]*(.*?)\s*\(Max\s*Marks:\s*(\d+)\)", re.DOTALL | re.IGNORECASE)
    matches = question_pattern.findall(text)
    
    return [(f"Q{qno}", qtext.strip(), float(max_mark)) for qno, qtext, max_mark in matches]

# -------------------------
# GPT Scoring Logic
# -------------------------
def score_answer(answer, rubric_text, max_mark):
    """
    Scores an answer based on a given rubric and maximum marks.
    Prioritizes GPT-4 for scoring; falls back to SentenceTransformer similarity if GPT-4 fails.

    Args:
        answer (str): The student's answer text.
        rubric_text (str): The specific rubric text for the question.
        max_mark (float): The maximum possible marks for the question.

    Returns:
        float: The calculated score for the answer.
    """
    stripped = answer.strip()
    if not stripped or len(stripped.split()) < 3 or stripped.lower() in ["n/a", "none", "no", "-"]:
        return 0

    prompt = f"""
You are an evaluator. Evaluate the answer strictly based on the rubric and assign marks accordingly. The rubric contains detailed weightages (100%, 75%, 50%, 25%, 0%). Do not provide explanation. Only return a number.

Rubric:
{rubric_text}

Answer:
{answer}

Give only a number out of {max_mark}.
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        score_str = response.choices[0].message.content.strip()

        if "openai_api_working_confirmed" not in st.session_state:
            st.success("🎉 OpenAI API key is correctly configured and working correctly!")
            st.session_state.openai_api_working_confirmed = True

        return float(score_str)
    except Exception as e:
        st.warning(f"GPT scoring failed for an answer (likely API key issue or rate limit): {e}. Falling back to similarity scoring.")
        
        emb_answer = model.encode(answer, convert_to_tensor=True)
        emb_rubric = model.encode(rubric_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_answer, emb_rubric).item()

        if similarity >= 0.9:
            return max_mark
        elif similarity >= 0.75:
            return 0.75 * max_mark
        elif similarity >= 0.5:
            return 0.5 * max_mark
        elif similarity >= 0.35:
            return 0.25 * max_mark
        else:
            return 0

# -------------------------
# Login Function
# -------------------------
def login():
    """
    Handles user login using Streamlit secrets for credentials.
    """
    st.title("🔐 Answer Sheet Evaluator - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if "credentials" in st.secrets and username in st.secrets["credentials"] and st.secrets["credentials"][username] == password:
            st.session_state.authenticated = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

# -------------------------
# Logout Function
# -------------------------
def logout():
    """
    Logs out the user by clearing authentication status.
    """
    st.session_state.authenticated = False
    st.info("You have been logged out.")
    st.rerun()

# -------------------------
# Session State Initialization
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "openai_api_working_confirmed" not in st.session_state:
    st.session_state.openai_api_working_confirmed = False

# -------------------------
# Main Application Flow
# -------------------------

# Inject custom CSS for the logo and copyright at the very beginning
inject_custom_css()

if not st.session_state.authenticated:
    login()
    st.stop()

# If authenticated, proceed with the main app
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure it.")
    st.stop()

# Add logout button to the sidebar
with st.sidebar:
    st.header("Navigation")
    if st.button("Logout", key="logout_button"):
        logout()
    # You can add more sidebar elements here if needed later


st.title("📄 Dynamic Answer Sheet Evaluation Dashboard")

st.header("Step 1: Upload Question Paper (Rubric)")
rubric_file = st.file_uploader("Upload Rubric File (.docx or .pdf)", type=["docx", "pdf"])

rubric_text = ""
question_blocks = []

if rubric_file:
    if rubric_file.name.endswith(".pdf"):
        rubric_text = extract_text_from_pdf(rubric_file)
    else:
        rubric_text = extract_text_from_docx(rubric_file)
    
    if rubric_text:
        question_blocks = parse_rubric_questions(rubric_text)

if not rubric_text or not question_blocks:
    st.warning("Please upload a valid rubric. Ensure questions are formatted like 'Q1: ... (Max Marks: 5)'.")
    st.stop()

st.header("Step 2: Upload Student Answer Sheets")
files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

results = {}
questions = [q for q, _, _ in question_blocks]
max_marks_list = [m for _, _, m in question_blocks]
question_rubric_map = {q: text for q, text, _ in question_blocks}
max_marks_map = {q: m for q, _, m in question_blocks}

if rubric_text and files:
    st.header("Step 3: Evaluation Results")

    for file in files:
        name = os.path.splitext(file.name)[0]
        
        if file.name.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            content = extract_text_from_docx(file)
        
        answers_split = re.split(r"Q(\d+)\s*Answer:", content, flags=re.IGNORECASE)
        
        question_ans_map = {}
        if len(answers_split) > 1:
            for i in range(1, len(answers_split), 2):
                if (i + 1) < len(answers_split):
                    q_no = f"Q{answers_split[i].strip()}"
                    ans = answers_split[i+1].strip()
                    question_ans_map[q_no] = ans
                else:
                    st.warning(f"Could not find answer content for Q{answers_split[i].strip()} in {name}. It might be the last question with no content following.")

        student_scores = []
        for q in questions:
            ans = question_ans_map.get(q, "")
            marks = score_answer(ans, question_rubric_map[q], max_marks_map[q])
            student_scores.append(marks)

        results[name] = student_scores

    df_scores = pd.DataFrame(results, index=questions)
    df_scores.insert(0, "Max Marks", max_marks_list)
    df_scores.index.name = "Q. No"
    
    total_row = [sum(max_marks_list)] + [df_scores[col].sum() for col in df_scores.columns if col != "Max Marks"]
    df_scores.loc["Total"] = total_row

    st.markdown("### 📊 Final Evaluation Table")
    st.dataframe(df_scores)

    def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        df.to_excel(writer, index=True, sheet_name='Evaluation')
        writer.close()
        return output.getvalue()

    excel_data = to_excel(df_scores)
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="evaluation_report.xlsx">📥 Download Excel Report</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    st.info("Upload rubric and answer sheets to begin evaluation.")
