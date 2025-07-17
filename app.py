import streamlit as st
import openai
import pandas as pd
import PyPDF2
import io
import re

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------- Helper Functions ----------

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def normalize_question_keys(text):
    questions = {}
    current_q = None
    for line in text.splitlines():
        line = line.strip()
        # Match Q1, Q1., q1, etc.
        match = re.match(r"^(Q?)(\d{1,2})[.:)]?\s", line, re.IGNORECASE)
        if match:
            qnum = f"Q{match.group(2)}"
            current_q = qnum
            questions[current_q] = line
        elif current_q:
            questions[current_q] += " " + line
    return questions

def extract_score(text):
    match = re.search(r"\b([0-5])\b", text)
    return int(match.group(1)) if match else 0

def evaluate_answer_with_gpt(question, answer):
    prompt = f"""
You are an academic evaluator for a university exam. You are given a question, its scoring rubric, and a student's answer.

Your task is to assign a score out of 5 marks strictly based on the rubric.

Do not be lenient or add explanation. Just follow the rubric.

---

Question:
{question}

Rubric (scoring guide):
- 5 marks: All key elements fully explained with appropriate examples.
- 4 marks: All elements explained, but examples are missing or partially relevant.
- 3 marks: At least 50% of the key elements explained.
- 2 marks: Only a few points or vague explanation.
- 1 mark: Minimal relevant content or vague/incorrect concepts.
- 0 marks: Not attempted or completely irrelevant answer.

---

Student's Answer:
{answer}

---

Only return the score as a number from 0 to 5. Do not include any other explanation.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a strict, objective academic evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = response['choices'][0]['message']['content'].strip()
        return extract_score(result)
    except Exception as e:
        st.warning(f"Evaluation failed: {e}")
        return 0

# ---------- Streamlit App ----------

st.set_page_config(page_title="Automated Answer Sheet Evaluator")
st.title("📊 Automated Answer Sheet Evaluator")

# Step 1: Upload Question Paper (Rubric)
st.header("1. Upload Question Paper with Rubrics")
qpaper_file = st.file_uploader("Upload Question Paper (PDF)", type=["pdf"])

# Step 2: Upload Student Answer Sheets
st.header("2. Upload Student Answer Sheets")
answer_files = st.file_uploader("Upload Answer Sheets (PDF)", type=["pdf"], accept_multiple_files=True)

# Process files
if qpaper_file and answer_files:
    with st.spinner("Processing question paper..."):
        qpaper_text = extract_text_from_pdf(qpaper_file)
        questions = normalize_question_keys(qpaper_text)

    # Create results table
    result_df = pd.DataFrame(columns=["Question No", "Max Marks"] + [f.name for f in answer_files])
    result_df["Question No"] = list(questions.keys())
    result_df["Max Marks"] = 5

    with st.spinner("Evaluating answers..."):
        for idx, (qno, qtext) in enumerate(questions.items()):
            for f in answer_files:
                answer_text = extract_text_from_pdf(f)
                answers = normalize_question_keys(answer_text)
                student_answer = answers.get(qno, "")
                score = evaluate_answer_with_gpt(qtext, student_answer)
                result_df.at[idx, f.name] = score

    # Add total row
    total_row = ["Total", ""] + [result_df[col][:-1].astype(int).sum() for col in result_df.columns[2:]]
    result_df.loc[len(result_df)] = total_row

    st.header("📋 Final Evaluation Table")
    st.dataframe(result_df, use_container_width=True)

    # Download option
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name='evaluation_results.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload both the question paper and at least one answer sheet.")
