import streamlit as st
import openai
import pandas as pd
import PyPDF2
import io

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ---------- Helper Functions ----------

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_questions_from_text(text):
    questions = {}
    current_q = None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("q") and "." in line:
            current_q = line.split('.')[0].strip().upper()
            questions[current_q] = line
        elif current_q:
            questions[current_q] += " " + line
    return questions

def evaluate_answer_with_gpt(question, rubric, answer):
    prompt = f"""
You are an exam evaluator.

Below is a question, its rubric, and a student's answer.

Evaluate the answer out of 5 marks based ONLY on the rubric. Return only a number between 0 and 5.

Question: {question}

Rubric: {rubric}

Student Answer: {answer}

Score (only number 0-5):
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
        return int(result) if result.isdigit() else 0
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
        questions = split_questions_from_text(qpaper_text)

    # Create empty dataframe to store results
    result_df = pd.DataFrame(columns=["Question No", "Max Marks"] + [f"{f.name}" for f in answer_files])
    result_df["Question No"] = list(questions.keys())
    result_df["Max Marks"] = 5

    with st.spinner("Evaluating all answer sheets..."):
        for idx, (qno, qtext) in enumerate(questions.items()):
            for f in answer_files:
                answer_text = extract_text_from_pdf(f)
                answers = split_questions_from_text(answer_text)
                student_answer = answers.get(qno, "")
                score = evaluate_answer_with_gpt(qtext, "Evaluate based on how complete and relevant the answer is.", student_answer)
                result_df.at[idx, f.name] = score

    # Add Total row
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
