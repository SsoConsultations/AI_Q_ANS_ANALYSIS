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
    # Updated regex: `[^A-Za-z0-9\n]*` is more flexible for the separator between Q# and question text.
    question_pattern = re.compile(r"Q(\d+)\s*[^A-Za-z0-9\n]*(.*?)\s*\(Max\s*Marks:\s*(\d+)\)", re.DOTALL | re.IGNORECASE)
    matches = question_pattern.findall(text)
    
    # Debug output removed as per user request
    # st.write("Regex matches found for rubric questions (for debugging):", matches) 
    
    # Convert captured max_mark to float and return the list of tuples.
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
    # Initial check for very short or placeholder answers
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
        # Attempt to score using OpenAI's GPT-4
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, # Set to 0 for deterministic output
        )
        score_str = response.choices[0].message.content.strip()

        # Display success message only for the first successful API call in the session
        # This confirms the OpenAI API key is working.
        if "openai_api_working_confirmed" not in st.session_state:
            st.success("🎉 OpenAI API key is correctly configured and working correctly!")
            st.session_state.openai_api_working_confirmed = True

        return float(score_str)
    except Exception as e:
        # Fallback to SentenceTransformer similarity if GPT-4 API call fails
        st.warning(f"GPT scoring failed for an answer (likely API key issue or rate limit): {e}. Falling back to similarity scoring.")
        
        emb_answer = model.encode(answer, convert_to_tensor=True)
        emb_rubric = model.encode(rubric_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_answer, emb_rubric).item()

        # Map similarity score to marks based on predefined thresholds
        if similarity >= 0.9:
            return max_mark # Excellent
        elif similarity >= 0.75:
            return 0.75 * max_mark # Good
        elif similarity >= 0.5:
            return 0.5 * max_mark # Acceptable
        elif similarity >= 0.35:
            return 0.25 * max_mark # Poor
        else:
            return 0 # Unacceptable

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
        # Check if username exists in secrets and password matches
        if "credentials" in st.secrets and username in st.secrets["credentials"] and st.secrets["credentials"][username] == password:
            st.session_state.authenticated = True
            st.success("Login successful")
            # Clear inputs for security
            st.rerun() # Rerun to clear login form and show main app
        else:
            st.error("Invalid username or password")

# -------------------------
# Session State Initialization
# -------------------------
# Initialize authentication status if not already set
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
# Initialize flag for OpenAI API confirmation message
if "openai_api_working_confirmed" not in st.session_state:
    st.session_state.openai_api_working_confirmed = False

# -------------------------
# Authentication Check
# -------------------------
# If not authenticated, display the login page and stop execution
if not st.session_state.authenticated:
    login()
    st.stop() # Stop further execution until authenticated

# -------------------------
# Initialize OpenAI API Key
# -------------------------
# This line will only be reached if the user is authenticated
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure it.")
    st.stop() # Stop if API key is not configured

# -------------------------
# Streamlit Application Start
# -------------------------
st.title("📄 Dynamic Answer Sheet Evaluation Dashboard")

st.header("Step 1: Upload Question Paper (Rubric)")
rubric_file = st.file_uploader("Upload Rubric File (.docx or .pdf)", type=["docx", "pdf"])

rubric_text = ""
question_blocks = []

if rubric_file:
    # Extract text based on file type
    if rubric_file.name.endswith(".pdf"):
        rubric_text = extract_text_from_pdf(rubric_file)
    else:
        rubric_text = extract_text_from_docx(rubric_file)
    
    # If text was successfully extracted, parse the questions
    if rubric_text:
        # Debug output removed as per user request
        # st.text_area("Extracted Text from Rubric File (for debugging)", rubric_text, height=300, key="rubric_text_debug")
        question_blocks = parse_rubric_questions(rubric_text)

# Check if rubric text is available and questions were parsed successfully
if not rubric_text or not question_blocks:
    st.warning("Please upload a valid rubric. Ensure questions are formatted like 'Q1: ... (Max Marks: 5)'.")
    st.stop() # Stop if rubric is not valid

# -------------------------
# Upload Answer Sheets
# -------------------------
st.header("Step 2: Upload Student Answer Sheets")
files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# -------------------------
# Evaluate Answers
# -------------------------
results = {}
# Prepare lists and maps for questions and their rubrics/max marks
questions = [q for q, _, _ in question_blocks]
max_marks_list = [m for _, _, m in question_blocks]
question_rubric_map = {q: text for q, text, _ in question_blocks}
max_marks_map = {q: m for q, _, m in question_blocks}

if rubric_text and files: # Proceed only if rubric and student files are uploaded
    st.header("Step 3: Evaluation Results")

    for file in files:
        name = os.path.splitext(file.name)[0] # Get student name from file name
        # Debug output removed as per user request
        # st.subheader(f"Debugging {name}") 

        # Extract content from student answer sheet
        if file.name.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            content = extract_text_from_docx(file)
        
        # Debug output removed as per user request
        # st.text_area(f"Extracted Text from {name} (for debugging)", content, height=300, key=f"student_text_debug_{name}")

        # Split the content into individual answers based on "Q# Answer:" pattern
        # The regex handles optional spaces and is case-insensitive for "Answer"
        answers_split = re.split(r"Q(\d+)\s*Answer:", content, flags=re.IGNORECASE)
        
        question_ans_map = {}
        # The re.split pattern will put any text *before* the first "Q# Answer:" into answers_split[0].
        # Subsequent elements will be alternating question numbers and their answers.
        if len(answers_split) > 1:
            for i in range(1, len(answers_split), 2):
                if (i + 1) < len(answers_split): # Ensure there's an answer chunk after the Q number
                    q_no = f"Q{answers_split[i].strip()}" # Format as "Q1", "Q2", etc.
                    ans = answers_split[i+1].strip() # Get the answer text
                    question_ans_map[q_no] = ans
                else:
                    # Log a warning if an answer chunk is missing for a question number
                    st.warning(f"Could not find answer content for Q{answers_split[i].strip()} in {name}. It might be the last question with no content following.")

        # Debug output removed as per user request
        # st.write(f"Parsed Answers for {name} (for debugging):", question_ans_map)

        student_scores = []
        # Iterate through all questions identified in the rubric
        for q in questions:
            ans = question_ans_map.get(q, "") # Get the answer, default to empty string if not found
            # Debug output removed as per user request
            # st.text(f"Scoring {q} for {name}: Answer length = {len(ans.split())} words.")
            marks = score_answer(ans, question_rubric_map[q], max_marks_map[q])
            student_scores.append(marks)

        results[name] = student_scores

    # -------------------------
    # Create and Display Score Table
    # -------------------------
    df_scores = pd.DataFrame(results, index=questions)
    df_scores.insert(0, "Max Marks", max_marks_list) # Add Max Marks column
    df_scores.index.name = "Q. No" # Set index name
    
    # Calculate total row
    total_row = [sum(max_marks_list)] + [df_scores[col].sum() for col in df_scores.columns if col != "Max Marks"]
    df_scores.loc["Total"] = total_row # Add total row to DataFrame

    st.markdown("### 📊 Final Evaluation Table")
    st.dataframe(df_scores)

    # -------------------------
    # Excel Download Functionality
    # -------------------------
    def to_excel(df):
        """
        Converts a Pandas DataFrame to an Excel file in memory.

        Args:
            df (pd.DataFrame): The DataFrame to convert.

        Returns:
            bytes: The Excel file content as bytes.
        """
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        df.to_excel(writer, index=True, sheet_name='Evaluation')
        writer.close() # Important to close the writer
        return output.getvalue()

    excel_data = to_excel(df_scores)
    b64 = base64.b64encode(excel_data).decode() # Encode to base64 for download link
    # Create a download link for the Excel file
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="evaluation_report.xlsx">📥 Download Excel Report</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    # Initial message if no files are uploaded yet
    st.info("Upload rubric and answer sheets to begin evaluation.")
