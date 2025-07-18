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
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------
# Extract text from PDF
# -------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

# -------------------------
# Extract text from DOCX
# -------------------------
def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

# -------------------------
# Extract questions from rubric with default 5 marks
# -------------------------
def parse_rubric_questions(text):
    question_pattern = re.compile(r"Q(\d+)[.:\)]\s*(.*?)\s*Marks\s*\(?\s*5\s*\)?", re.DOTALL)
    matches = question_pattern.findall(text)
    return [(f"Q{qno}", qtext.strip(), 5.0) for qno, qtext in matches]

# -------------------------
# GPT Scoring Logic
# -------------------------
def score_answer(answer, rubric_text, max_mark):
    stripped = answer.strip()
    if not stripped or len(stripped.split()) < 5 or stripped.lower() in ["n/a", "none", "no", "-"]:
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
        return float(score_str)
    except:
        emb_answer = model.encode(answer, convert_to_tensor=True)
        emb_rubric = model.encode(rubric_text, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(emb_answer, emb_rubric).item()

        if similarity < 0.35:
            return 0
        elif similarity >= 0.9:
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
# Login
# -------------------------
def login():
    st.title("🔐 Answer Sheet Evaluator - Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in st.secrets["credentials"] and st.secrets["credentials"][username] == password:
            st.session_state.authenticated = True
            st.success("Login successful")
        else:
            st.error("Invalid username or password")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login()
    st.stop()

# -------------------------
# Initialize OpenAI
# -------------------------
openai.api_key = st.secrets["openai"]["api_key"]

# -------------------------
# App Start
# -------------------------
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
    question_blocks = parse_rubric_questions(rubric_text)

if not rubric_text or not question_blocks:
    st.warning("Please upload a valid rubric with Q1, Q2... and 'Marks (5)' in each question.")
    st.stop()

# -------------------------
# Upload Answer Sheets
# -------------------------
st.header("Step 2: Upload Student Answer Sheets")
files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# -------------------------
# Evaluate Answers
# -------------------------
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

        answers_split = re.split(r"Q(\d+)[.:\)\n]", content)
        question_ans_map = {}
        for i in range(1, len(answers_split), 2):
            q_no = f"Q{answers_split[i]}"
            ans = answers_split[i+1].strip()
            question_ans_map[q_no] = ans

        student_scores = []
        for q in questions:
            ans = question_ans_map.get(q, "")
            marks = score_answer(ans, question_rubric_map[q], max_marks_map[q])
            student_scores.append(marks)

        results[name] = student_scores

    # Create Score Table
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
