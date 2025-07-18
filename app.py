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
# NEW: Parse Questions from QUESTION_PAPER.docx
# -------------------------
def parse_question_paper_file(text):
    """
    Parses text from QUESTION_PAPER.docx to extract question number,
    clean question text (removing Max Marks), and Max Marks.
    Expected format: Q#: Question text (Max Marks: #)

    Returns a dictionary: {'Q#': {'text': 'clean question text', 'max_marks': float}, ...}
    """
    questions_data = {}
    # Regex to capture Q#, question text (including Max Marks part), and Max Marks value
    # We capture the entire Q and the Max Marks part to remove it later for clean text.
    question_pattern = re.compile(
        r"Q(\d+):\s*(.*?)(?:\s*\(Max\s*Marks:\s*(\d+)\))?",
        re.DOTALL | re.IGNORECASE
    )
    # The (?:...) non-capturing group for Max Marks makes it optional
    # and the outer capturing group for question text includes it if present.

    # Find all matches
    matches = question_pattern.finditer(text)

    for match in matches:
        q_num = f"Q{match.group(1)}"
        full_question_text = match.group(2).strip()
        max_marks = match.group(3) # This will be the numerical part or None

        # Clean the question text: remove the (Max Marks: X) part if it exists
        clean_question_text = re.sub(r'\s*\(Max\s*Marks:\s*\d+\)', '', full_question_text, flags=re.IGNORECASE).strip()

        # Convert max_marks to float, default to 0 or raise error if crucial
        try:
            max_marks_float = float(max_marks) if max_marks else 0.0 # Default to 0 if not found, for validation
        except ValueError:
            st.error(f"Could not parse Max Marks for {q_num} from Question Paper. Please ensure it's a number.")
            max_marks_float = 0.0 # Or raise an error to stop execution

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

# -------------------------
# Authentication Check
# -------------------------
if not st.session_state.authenticated:
    inject_logo_and_copyright_css()
    login()
    st.stop()

# -------------------------
# Initialize OpenAI API Key (only if authenticated)
# -------------------------
try:
    openai.api_key = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure it in .streamlit/secrets.toml")
    st.stop()

# -------------------------
# Streamlit Application Start (Main App)
# -------------------------
inject_logo_and_copyright_css()

st.title("Dynamic Answer Sheet Evaluation Dashboard")

# -------------------------
# Logout Button in Sidebar
# -------------------------
with st.sidebar:
    st.header("Navigation")
    if st.button("Logout"):
        st.session_state.authenticated = False
        # Clear sensitive session data on logout
        st.session_state.openai_api_working_confirmed = False
        st.session_state.all_questions_data = None
        st.rerun()

# -------------------------
# Step 1: Upload Question Paper and Rubric
# -------------------------
st.header("Step 1: Upload Question Paper & Rubric")

col1, col2 = st.columns(2) # Use columns for side-by-side uploaders

with col1:
    question_paper_file = st.file_uploader("Upload Question Paper (.docx or .pdf)", type=["docx", "pdf"], key="qp_uploader")

with col2:
    rubric_file = st.file_uploader("Upload Rubric File (.docx or .pdf)", type=["docx", "pdf"], key="rubric_uploader")

# Process and validate files if both are uploaded
if question_paper_file and rubric_file:
    qp_text = ""
    rubric_text = ""

    # Extract text from Question Paper
    if question_paper_file.name.endswith(".pdf"):
        qp_text = extract_text_from_pdf(question_paper_file)
    else:
        qp_text = extract_text_from_docx(question_paper_file)

    # Extract text from Rubric File
    if rubric_file.name.endswith(".pdf"):
        rubric_text = extract_text_from_pdf(rubric_file)
    else:
        rubric_text = extract_text_from_docx(rubric_file)

    if qp_text and rubric_text:
        # Parse data from both files
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
        all_questions_data = {} # This will store the merged and validated data
        
        # Get all unique question numbers from both files
        all_q_nums = sorted(list(set(qp_questions_data.keys()).union(set(parsed_rubrics_data.keys()))))

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
            
            # If no errors for this specific question, merge the data
            # Use max_marks from rubric as it's the primary source in the new setup
            all_questions_data[q_num] = {
                'question_text': qp_info['text'], # Cleaned text from QP
                'rubric_text': rubric_info['rubric_text'], # Detailed rubric text
                'max_marks': rubric_info['max_marks'] # Validated max marks
            }
        
        # --- Display Validation Results ---
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            st.warning("Please add appropriate document(s) to resolve the discrepancies.")
            st.session_state.all_questions_data = None # Clear data on validation failure
            st.stop() # Stop further execution
        else:
            st.success("Documents validated successfully! All question counts and Max Marks match.")
            st.session_state.all_questions_data = all_questions_data # Store merged data in session state
            # Force a rerun to proceed with the app, ensuring the state is updated
            st.rerun() 
    elif not qp_text or not rubric_text:
        st.warning("Could not extract text from one or both uploaded files. Please check the file formats.")
        st.session_state.all_questions_data = None
        st.stop()
else:
    # Initial message if no files are uploaded yet, or only one is uploaded
    if not st.session_state.all_questions_data:
        st.info("Please upload both the Question Paper and the Rubric File to begin validation.")
        st.stop() # Stop if files are not uploaded yet or not validated

# If we reach here, it means files were either just validated or already validated in a previous rerun
# Proceed with the rest of the application using st.session_state.all_questions_data
if st.session_state.all_questions_data is None or len(st.session_state.all_questions_data) == 0:
    st.error("Error: Validated question and rubric data is missing. Please re-upload documents.")
    st.stop() # Ensure we stop if data is somehow missing

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

    for file in files:
        name = os.path.splitext(file.name)[0] # Get student name from file name

        # Extract content from student answer sheet
        if file.name.endswith(".pdf"):
            content = extract_text_from_pdf(file)
        else:
            content = extract_text_from_docx(file)
        
        # Split the content into individual answers based on "Q# Answer:" pattern
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
        # Iterate through all questions identified in the validated data
        for q in all_questions_keys:
            ans = question_ans_map.get(q, "") # Get the answer, default to empty string if not found
            # Use the merged data for rubric and max marks
            marks = score_answer(ans, question_rubric_map[q], max_marks_list[questions_for_df.index(q)])
            student_scores.append(marks)

        results[name] = student_scores

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
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="evaluation_report.xlsx">Download Excel Report</a>'
    st.markdown(href, unsafe_allow_html=True)
else:
    # Initial message if no student files are uploaded yet, and we have QP/Rubric data
    if st.session_state.all_questions_data:
        st.info("Upload student answer sheets to begin evaluation.")
    # Else, the initial message from above (upload QP/Rubric) will be shown
