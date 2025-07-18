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
        st.error(f"Error: Image file not found at '{path}'. Please ensure it exists.")
        return None
    except Exception as e:
        st.error(f"Error reading image file '{path}': {e}")
        return None

# -------------------------
# Custom CSS for logo and copyright
# -------------------------
def inject_logo_and_copyright_css(logo_path="SsoLogo.jpg"):
    """
    Injects custom CSS to display a logo at the top right corner
    and a copyright notice at the bottom center.
    """
    logo_base64 = get_image_as_base64(logo_path)

    # Common CSS for the entire app to adjust main content padding for sidebar
    # This ensures the main content doesn't overlap the sidebar or feel too cramped
    st.markdown("""
        <style>
        .block-container {
            padding-left: 1rem; /* Adjust padding if sidebar pushes content too much */
            padding-right: 1rem;
            padding-top: 2rem; /* Add some top padding as logo is fixed */
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown(
        f"""
        <style>
        .top-right-logo-container {{
            position: fixed;
            top: 20px;
            right: 20px;
            width: 120px;
            height: auto;
            z-index: 1000; /* Ensure it's above other elements */
        }}
        .top-right-logo-container img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .bottom-center-copyright {{
            position: fixed;
            bottom: 10px;
            left: 50%; /* Start at 50% from the left */
            transform: translateX(-50%); /* Move back by half its own width to center */
            font-size: 0.8em;
            color: #888888; /* Light gray color */
            z-index: 999; /* Below logo */
            white-space: nowrap; /* Prevent wrapping if too long */
        }}
        </style>
        """
        + (f"""
        <div class="top-right-logo-container">
            <img src="data:image/jpeg;base64,{logo_base64}" alt="Company Logo">
        </div>
        """ if logo_base64 else "") +
        """
        <div class="bottom-center-copyright">
            ©copyright SSO Consultants
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# Initialize NLP Model (Cached with spinner feedback)
# -------------------------
@st.cache_resource
def load_model_with_feedback():
    """
    Loads and caches the SentenceTransformer model 'all-MiniLM-L6-v2' with feedback.
    This model is used for calculating semantic similarity between text snippets.
    """
    st.info("Starting to load NLP model. This is a large model and may take a moment on the first run...")
    with st.spinner("Loading NLP model..."):
        model = SentenceTransformer('all-MiniLM-L6-v2')
    st.success("NLP model loaded successfully!")
    return model

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
        # For file uploaders, file.seek(0) ensures we read from the beginning
        file.seek(0)
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
        # For file uploaders, file.seek(0) ensures we read from the beginning
        file.seek(0)
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

# -------------------------
# NEW: Parse Questions from QUESTION_PAPER.docx (UPDATED FIX)
# -------------------------
def parse_question_paper_file(text):
    """
    Parses text from QUESTION_PAPER.docx to extract question number,
    clean question text (removing Max Marks), and Max Marks.
    Expected format: Q#: Question text (Max Marks: #) - can span lines.

    Returns a dictionary: {'Q#': {'text': 'clean question text', 'max_marks': float}, ...}
    """
    questions_data = {}
    # Regex to capture Q#, and the entire rest of the line/block for the question
    # We use a non-greedy match (.*?) followed by a positive lookahead to ensure we stop at the next question or end of string.
    # This captures the full question block including the (Max Marks: #) part.
    question_block_pattern = re.compile(
        r"(Q(\d+):\s*.*?)(?=\n*Q\d+:|\Z)", # Capture until next Q# followed by :, potentially with newlines, or end of string
        re.DOTALL | re.IGNORECASE
    )

    matches = question_block_pattern.finditer(text)

    for match in matches:
        full_question_block = match.group(1).strip() # This is "Q#: Question text (Max Marks: #)\n(Max Marks: #)"
        q_num = f"Q{match.group(2)}"

        # Now, separately extract Max Marks from the full_question_block
        max_marks_match = re.search(r"\(Max\s*Marks:\s*(\d+)\)", full_question_block, re.IGNORECASE)
        max_marks_float = 0.0
        if max_marks_match:
            try:
                max_marks_float = float(max_marks_match.group(1))
            except ValueError:
                st.error(f"Could not parse Max Marks for {q_num} from Question Paper (value error). Please ensure it's a number.")

        # Clean the question text: remove the "Q#:" prefix and the "(Max Marks: X)" part
        clean_question_text = re.sub(r"Q\d+:\s*", "", full_question_block, flags=re.IGNORECASE).strip()
        clean_question_text = re.sub(r'\s*\(Max\s*Marks:\s*\d+\)', '', clean_question_text, flags=re.IGNORECASE).strip()

        questions_data[q_num] = {
            'text': clean_question_text,
            'max_marks': max_marks_float
        }
    return questions_data


# -------------------------
# NEW: Parse Rubrics from RUBRIC.docx
# -------------------------
def parse_rubric_file(text):
    """
    Parses text from RUBRIC.docx to extract rubric number, detailed rubric text, and Max Marks.
    Expected format:
    Q# Rubric:
    Max Marks: #
    Detailed rubric content...

    Returns a dictionary: {'Q#': {'rubric_text': 'full rubric content', 'max_marks': float}, ...}
    """
    rubrics_data = {}
    # Split text by "Q# Rubric:" to get individual rubric blocks
    # Using findall and then processing each block allows for flexible content
    rubric_blocks = re.findall(r"(Q(\d+)\s*Rubric:.*?)(?=Q\d+\s*Rubric:|$)", text, re.DOTALL | re.IGNORECASE)

    for block_match in rubric_blocks:
        full_block = block_match[0] # The entire block starting from "Q# Rubric:"
        q_num = f"Q{block_match[1]}" # The extracted question number

        # Extract Max Marks from within the block
        max_marks_match = re.search(r"Max\s*Marks:\s*(\d+)", full_block, re.IGNORECASE)
        max_marks_float = 0.0
        if max_marks_match:
            try:
                max_marks_float = float(max_marks_match.group(1))
            except ValueError:
                st.error(f"Could not parse Max Marks for {q_num} from Rubric File. Please ensure it's a number.")

        # Extract the detailed rubric text, excluding "Q# Rubric:" and "Max Marks: #" lines
        rubric_text_lines = []
        lines = full_block.split('\n')
        for line in lines:
            if not re.search(r"Q\d+\s*Rubric:|Max\s*Marks:\s*\d+", line, re.IGNORECASE):
                rubric_text_lines.append(line.strip())
        detailed_rubric_text = "\n".join(filter(None, rubric_text_lines)).strip() # Remove empty lines

        rubrics_data[q_num] = {
            'rubric_text': detailed_rubric_text,
            'max_marks': max_marks_float
        }
    return rubrics_data

# -------------------------
# Score Answer using OpenAI (GPT-4) or SentenceTransformer
# -------------------------
def score_answer(student_answer, ideal_rubric, max_marks):
    """
    Scores a student's answer against an ideal rubric using OpenAI's GPT-4,
    with a fallback to SentenceTransformer for semantic similarity.

    Args:
        student_answer (str): The text of the student's answer.
        ideal_rubric (str): The detailed rubric for the question.
        max_marks (float): The maximum marks for the question.

    Returns:
        float: The score awarded to the student's answer.
    """
    if not student_answer.strip():
        return 0.0 # Return 0 if student answer is empty

    # Try scoring with OpenAI GPT-4
    if st.session_state.openai_api_working_confirmed:
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that evaluates student answers based on a given rubric. Provide a score out of the maximum marks. Be objective and fair. Focus only on the content matching the rubric."},
                    {"role": "user", "content": f"Student Answer: {student_answer}\n\nRubric: {ideal_rubric}\n\nMax Marks: {max_marks}\n\nBased on the rubric, how many marks (out of {max_marks}) should the student get? Provide only the numerical score. If the answer does not align with the rubric at all, provide 0."}
                ],
                max_tokens=50,
                temperature=0.0 # Keep temperature low for factual scoring
            )
            score_text = response.choices[0].message.content.strip()
            # Attempt to extract a numerical score, even if extra text is present
            numerical_score_match = re.search(r'(\d+(\.\d+)?)', score_text)
            if numerical_score_match:
                gpt_score = float(numerical_score_match.group(1))
                # Ensure score is within bounds [0, max_marks]
                return max(0.0, min(gpt_score, max_marks))
            else:
                st.warning(f"GPT-4 did not return a clear numerical score for: '{student_answer[:50]}...'. Falling back to semantic similarity.")
                st.session_state.openai_api_working_confirmed = False # Assume API is not working as expected
        except openai.AuthenticationError:
            st.error("OpenAI API authentication failed. Please check your API key in Streamlit secrets.")
            st.session_state.openai_api_working_confirmed = False # Disable further API calls
        except openai.APITimeoutError:
            st.warning("OpenAI API call timed out. Falling back to semantic similarity.")
            st.session_state.openai_api_working_confirmed = False # Assume API is slow/down
        except openai.APIError as e:
            st.warning(f"OpenAI API error: {e}. Falling back to semantic similarity.")
            st.session_state.openai_api_working_confirmed = False # Assume API is not working

    # Fallback to SentenceTransformer semantic similarity
    try:
        # Load model (from cache)
        model_st = load_model_with_feedback()
        # Encode sentences to get their embeddings
        answer_embedding = model_st.encode(student_answer, convert_to_tensor=True)
        rubric_embedding = model_st.encode(ideal_rubric, convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_similarity = util.cos_sim(answer_embedding, rubric_embedding).item()

        # Scale similarity to marks
        # A simple linear scaling: similarity 0-1 maps to marks 0-max_marks
        # You might want to fine-tune this scaling based on empirical results
        semantic_score = cosine_similarity * max_marks
        return max(0.0, min(semantic_score, max_marks)) # Ensure score is within bounds
    except Exception as e:
        st.error(f"Error with SentenceTransformer: {e}")
        return 0.0


# -------------------------
# Login Function
# -------------------------
def login():
    """
    Handles user login using Streamlit secrets for credentials.
    """
    st.title("Answer Sheet Evaluator - Login")
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
# Session State Initialization
# -------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "openai_api_working_confirmed" not in st.session_state:
    st.session_state.openai_api_working_confirmed = False
# New session state for parsed document data
if "all_questions_data" not in st.session_state:
    st.session_state.all_questions_data = None
# New session state for initialization status
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# -------------------------
# Authentication Check
# -------------------------
if not st.session_state.authenticated:
    inject_logo_and_copyright_css()
    login()
    st.stop()

# -------------------------
# Streamlit Application Start (Main App)
# -------------------------
inject_logo_and_copyright_css()

st.title("Dynamic Answer Sheet Evaluation Dashboard")

# -------------------------
# Initial Setup / Loading (This section runs immediately after login)
# -------------------------
with st.sidebar:
    st.header("App Status")
    
    # Initialize flag for loading NLP model and OpenAI key
    if not st.session_state.initialized:
        # Load NLP model early with detailed feedback
        model_nlp = load_model_with_feedback() # Call once to load and show feedback

        # Initialize OpenAI API Key and check validity with detailed feedback
        st.info("Starting OpenAI API key verification...")
        try:
            openai.api_key = st.secrets["openai"]["api_key"]
            with st.spinner("Verifying OpenAI API key..."):
                try:
                    # Use a very small, cheap model for a quick check
                    openai.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=5
                    )
                    st.session_state.openai_api_working_confirmed = True
                    st.success("OpenAI API key confirmed!")
                except Exception as e:
                    st.warning(f"OpenAI API key may not be fully functional: {e}. Evaluation will fall back to local model.")
                    st.session_state.openai_api_working_confirmed = False
        except KeyError:
            st.error("OpenAI API key not found in Streamlit secrets. Please configure it in .streamlit/secrets.toml")
            st.session_state.openai_api_working_confirmed = False
        
        # Mark initialization as complete
        st.session_state.initialized = True
        st.rerun() # Rerun to display the main app sections after initialization
    else:
        # If already initialized, just show status (these messages come from session state/cache)
        st.success("App resources are ready!")
        # Ensure the model is loaded (from cache, so fast) for the main script
        model_nlp = load_model_with_feedback() 
        if st.session_state.openai_api_working_confirmed:
            st.success("OpenAI API key confirmed!")
        else:
            st.warning("OpenAI API key not functional, using local model.")


    st.divider() # Separator
    st.header("Navigation")
    if st.button("Logout"):
        st.session_state.authenticated = False
        # Clear sensitive session data on logout
        st.session_state.openai_api_working_confirmed = False
        st.session_state.all_questions_data = None
        st.session_state.initialized = False # Reset initialized state
        st.rerun()


# --- Only show the main app sections if initialization is complete ---
if not st.session_state.initialized:
    st.info("Please wait while the application initializes necessary models and API connections. This may take a moment.")
    st.stop() # Stop further execution until initialization is done

# -------------------------
# Step 1: Upload Question Paper and Rubric
# -------------------------
st.header("Step 1: Upload Question Paper & Rubric")

col1, col2 = st.columns(2) # Use columns for side-by-side uploaders

with col1:
    question_paper_file = st.file_uploader("Upload Question Paper (.docx or .pdf)", type=["docx", "pdf"], key="qp_uploader")

with col2:
    rubric_file = st.file_uploader("Upload Rubric File (.docx or .pdf)", type=["docx", "pdf"], key="rubric_uploader")

# Process and validate files if both are uploaded, or if they were already processed
if question_paper_file and rubric_file:
    # Only re-process if files have changed (simple check)
    if (st.session_state.get('last_qp_file_id') != question_paper_file.file_id or
        st.session_state.get('last_rubric_file_id') != rubric_file.file_id or
        st.session_state.all_questions_data is None): # Reprocess if data is not loaded

        qp_text = ""
        rubric_text = ""

        # Extract text from Question Paper
        with st.spinner("Extracting text from Question Paper..."):
            if question_paper_file.name.endswith(".pdf"):
                qp_text = extract_text_from_pdf(question_paper_file)
            else:
                qp_text = extract_text_from_docx(question_paper_file)

        # Extract text from Rubric File
        with st.spinner("Extracting text from Rubric File..."):
            if rubric_file.name.endswith(".pdf"):
                rubric_text = extract_text_from_pdf(rubric_file)
            else:
                rubric_text = extract_text_from_docx(rubric_file)

        if qp_text and rubric_text:
            # Parse data from both files
            with st.spinner("Parsing question paper and rubric data..."):
                qp_questions_data = parse_question_paper_file(qp_text)
                parsed_rubrics_data = parse_rubric_file(rubric_text)

            # --- Validation Logic ---
            st.subheader("Document Validation")
            validation_errors = []

            # 1. Check total number of questions/rubrics
            if len(qp_questions_data) != len(parsed_rubrics_data):
                validation_errors.append(
                    f"Mismatch in total number of questions: Question Paper has {len(qp_questions_data)} "
                    f"questions, while Rubric File has {len(parsed_rubrics_data)} rubrics."
                )

            # 2. Check Max Marks consistency for each question
            current_all_questions_data = {} # Temp dictionary for current validation run
            
            # Get all unique question numbers from both files, sorted numerically
            all_q_nums = sorted(list(set(qp_questions_data.keys()).union(set(parsed_rubrics_data.keys()))), key=lambda x: int(x[1:]))

            for q_num in all_q_nums:
                qp_info = qp_questions_data.get(q_num)
                rubric_info = parsed_rubrics_data.get(q_num)

                if not qp_info:
                    validation_errors.append(f"Question {q_num} found in Rubric File but not in Question Paper.")
                    continue
                if not rubric_info:
                    validation_errors.append(f"Question {q_num} found in Question Paper but not in Rubric File.")
                    continue

                # Compare Max Marks
                if qp_info['max_marks'] != rubric_info['max_marks']:
                    validation_errors.append(
                        f"Max Marks mismatch for {q_num}: Question Paper has {qp_info['max_marks']}, "
                        f"Rubric File has {rubric_info['max_marks']}."
                    )
                
                # If no errors for this specific question (yet), merge the data for this run
                current_all_questions_data[q_num] = {
                    'question_text': qp_info['text'], # Cleaned text from QP
                    'rubric_text': rubric_info['rubric_text'], # Detailed rubric text
                    'max_marks': rubric_info['max_marks'] # Validated max marks (from rubric if matching)
                }
            
            # --- Display Validation Results ---
            if validation_errors:
                for error in validation_errors:
                    st.error(error)
                st.warning("Please add appropriate document(s) to resolve the discrepancies.")
                st.session_state.all_questions_data = None # Clear data on validation failure
            else:
                st.success("Documents validated successfully! All question counts and Max Marks match.")
                st.session_state.all_questions_data = current_all_questions_data # Store merged data in session state
                # Store file IDs to prevent reprocessing on rerun
                st.session_state.last_qp_file_id = question_paper_file.file_id
                st.session_state.last_rubric_file_id = rubric_file.file_id
                st.rerun() # Force a rerun to display the next steps after successful validation
        elif not qp_text or not rubric_text:
            st.warning("Could not extract text from one or both uploaded files. Please check the file formats.")
            st.session_state.all_questions_data = None
    # If files haven't changed and data is already in session state, don't re-run processing
    else:
        st.success("Documents validated successfully! All question counts and Max Marks match.")
        # We don't need to rerun here, as the state is already set from a previous run.
        # This prevents an infinite rerun loop if files are already processed.

else: # If files are not yet uploaded or partially uploaded
    if st.session_state.all_questions_data is None: # Only show this message if no data is loaded
        st.info("Please upload both the Question Paper and the Rubric File to begin validation.")
        st.stop() # Stop further execution until files are uploaded and validated


# If we reach here, it means files were either just validated or already validated in a previous rerun
# Proceed with the rest of the application using st.session_state.all_questions_data
if st.session_state.all_questions_data is None or len(st.session_state.all_questions_data) == 0:
    st.error("Error: Validated question and rubric data is missing. Please re-upload documents.")
    st.stop() # Ensure we stop if data is somehow missing (e.g., after validation error)

# Prepare lists/maps from the validated and merged data
all_questions_keys = sorted(st.session_state.all_questions_data.keys(), key=lambda x: int(x[1:])) # Sort Q1, Q2, etc.
questions_for_df = [q for q in all_questions_keys]
max_marks_list = [st.session_state.all_questions_data[q]['max_marks'] for q in all_questions_keys]
question_rubric_map = {q: st.session_state.all_questions_data[q]['rubric_text'] for q in all_questions_keys}
question_text_map = {q: st.session_state.all_questions_data[q]['question_text'] for q in all_questions_keys} # Store clean Q text

# -------------------------
# Step 2: Upload Student Answer Sheets
# -------------------------
st.header("Step 2: Upload Student Answer Sheets")
files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

# -------------------------
# Evaluate Answers
# -------------------------
results = {}

if files: # Proceed only if student files are uploaded
    st.header("Step 3: Evaluation Results")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, file in enumerate(files):
        name = os.path.splitext(file.name)[0] # Get student name from file name
        status_text.text(f"Processing student: {name} ({i+1}/{len(files)})")

        # Extract content from student answer sheet
        with st.spinner(f"Extracting text from {name}'s answer sheet..."):
            if file.name.endswith(".pdf"):
                content = extract_text_from_pdf(file)
            else:
                content = extract_text_from_docx(file)
        
        # Split the content into individual answers based on "Q# Answer:" pattern
        answers_split = re.split(r"Q(\d+)\s*Answer:", content, flags=re.IGNORECASE)
        
        question_ans_map = {}
        if len(answers_split) > 1:
            for j in range(1, len(answers_split), 2):
                if (j + 1) < len(answers_split):
                    q_no = f"Q{answers_split[j].strip()}"
                    ans = answers_split[j+1].strip()
                    question_ans_map[q_no] = ans
                else:
                    st.warning(f"Could not find answer content for Q{answers_split[j].strip()} in {name}. It might be the last question with no content following.")

        student_scores = []
        # Iterate through all questions identified in the validated data
        for q_idx, q in enumerate(all_questions_keys):
            ans = question_ans_map.get(q, "") # Get the answer, default to empty string if not found
            # Use the merged data for rubric and max marks
            # Ensure the index is correct based on the sorted all_questions_keys
            status_text.text(f"Evaluating {name}'s answer to {q}...")
            marks = score_answer(ans, question_rubric_map[q], max_marks_list[questions_for_df.index(q)])
            student_scores.append(marks)

        results[name] = student_scores
        progress_bar.progress((i + 1) / len(files))

    status_text.text("All student answers processed!")
    st.success("Evaluation complete!")

    # -------------------------
    # Create and Display Score Table
    # -------------------------
    df_scores = pd.DataFrame(results, index=questions_for_df)
    df_scores.insert(0, "Max Marks", max_marks_list) # Add Max Marks column
    df_scores.index.name = "Q. No" # Set index name
    
    # Calculate total row
    total_row = [sum(max_marks_list)] + [df_scores[col].sum() for col in df_scores.columns if col != "Max Marks"]
    df_scores.loc["Total"] = total_row # Add total row to DataFrame

    st.markdown("### Final Evaluation Table")
    st.dataframe(df_scores)

    # -------------------------
    # Excel Download Functionality
    # ------------------------
    def to_excel(df):
        """
        Converts a Pandas DataFrame to an Excel file in memory.
        """
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='openpyxl')
        df.to_excel(writer, index=True, sheet_name='Evaluation')
        writer.close()
        return output.getvalue()

    excel_data = to_excel(df_scores)
    b64 = base66.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="evaluation_report.xlsx">Download Excel Report</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    # Initial message if no student files are uploaded yet, and we have QP/Rubric data
    if st.session_state.all_questions_data:
        st.info("Upload student answer sheets to begin evaluation.")
    # Else, the initial message from above (upload QP/Rubric) will be shown
