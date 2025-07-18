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
    question_pattern = re.compile(r"Q(\d+)[.:\)]\s*(.*?)\s*\(Max\s*Marks:\s*(\d+)\)", re.DOTALL)
    matches = question_pattern.findall(text)
    return [(f"Q{qno}", qtext.strip(), float(max_mark)) for qno, qtext, max_mark in matches]

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
    except Exception as e:
        st.warning(f"GPT scoring failed for an answer: {e}. Falling back to similarity scoring. Check your OpenAI API key and usage limits if this persists.") #
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
    
    if rubric_text:
        question_blocks = parse_rubric_questions(rubric_text)

if not rubric_text or not question_blocks:
    st.warning("Please upload a valid rubric. Ensure questions are formatted like 'Q1: ... (Max Marks: 5)'.")
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
        st.subheader(f"Debugging {name}") #

        if file.name.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            content = extract_text_from_docx(file)
        
        # DEBUG: Display extracted raw text
        st.text_area(f"Extracted Text from {name}", content, height=300) #

        # The regex for splitting student answers needs to be robust.
        # Let's refine it slightly to ensure it handles "Q# Answer:" and doesn't get confused by "Answer Sheet - Student_X"
        answers_split = re.split(r"Q(\d+)\s*Answer:", content, flags=re.IGNORECASE) #
        
        question_ans_map = {}
        # The split will put the text before the first Q# into answers_split[0]
        # We need to iterate from 1, checking pairs of (question_number, answer_text)
        if len(answers_split) > 1:
            for i in range(1, len(answers_split), 2):
                if (i + 1) < len(answers_split): # Ensure there's an answer chunk after the Q number
                    q_no = f"Q{answers_split[i].strip()}" # Strip to remove potential whitespace around the number
                    ans = answers_split[i+1].strip()
                    question_ans_map[q_no] = ans
                else:
                    # Handle case where the last Q has no content following it or split issue
                    st.warning(f"Could not find answer content for Q{answers_split[i].strip()} in {name}.") #


        # DEBUG: Display the parsed question_ans_map
        st.write(f"Parsed Answers for {name}:", question_ans_map) #


        student_scores = []
        for q in questions:
            ans = question_ans_map.get(q, "")
            # DEBUG: Display which answer is being scored for which question
            st.text(f"Scoring {q} for {name}: Answer length = {len(ans.split())} words.") #
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
