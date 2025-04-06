import os
import json
import base64
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain_groq import ChatGroq
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from langchain.agents.agent import AgentExecutor

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
google_app_password = os.getenv("GOOGLE_APP_PASSWORD")

# Initialize models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.2)

# --- Sidebar File Uploads / Input ---
st.sidebar.header("Upload Files / Enter Job Description")
# Instead of uploading a JD CSV, enter the details:
JD_FILE_PATH = r"C:\Users\meetu\Agentic AI - KrishNaik\MultiAgent_JobScreening\job_description.csv"
job_title = st.sidebar.text_input("Job Title")
job_description = st.sidebar.text_area("Job Description (max 500 words)", max_chars=3000)
if st.sidebar.button("Save Job Description"):
    if job_title and job_description:
        if len(job_description.split()) > 500:
            st.sidebar.error("Job Description exceeds 500 words.")
        else:
            # Check if the file exists; if so, append the new job description
            if os.path.exists(JD_FILE_PATH):
                df_existing = pd.read_csv(JD_FILE_PATH, encoding="ISO-8859-1")
                df_new = pd.DataFrame({"Job Title": [job_title], "Job Description": [job_description]})
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_final = pd.DataFrame({"Job Title": [job_title], "Job Description": [job_description]})
            df_final.to_csv(JD_FILE_PATH, index=False, encoding="ISO-8859-1")
            st.sidebar.success("Job Description saved successfully.")
    else:
        st.sidebar.error("Please enter both a Job Title and a Job Description.")

# Upload CV PDF
uploaded_cv = st.sidebar.file_uploader("Upload CV PDF", type=["pdf"])
CV_FOLDER = r"C:\Users\meetu\Agentic AI - KrishNaik\MultiAgent_JobScreening\CVs1"
if uploaded_cv is not None:
    cv_file_path = os.path.join(CV_FOLDER, uploaded_cv.name)
    with open(cv_file_path, "wb") as f:
        f.write(uploaded_cv.getbuffer())
    st.sidebar.success("CV uploaded successfully.")

def get_output(result):
    if isinstance(result, dict):
        return result.get("output", str(result))
    return result

def embed_and_store(chunks, name):
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    texts = [str(chunk) for chunk in chunks]
    vectordb = FAISS.from_texts(texts, embedding_model)
    vectordb.save_local(f"faiss_index/{name}")
    print(f"✅ Stored embeddings for {name} in faiss_index/{name}")

def summarize_job_description(jd_text):
    st.info("Summarizing the Job Description...")
    prompt = PromptTemplate(
        input_variables=["jd"],
        template="""
Summarize the following job description into key responsibilities and qualifications:

{jd}

Return a concise summary.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({"jd": jd_text})

jd_tools = [Tool(name="Summarize JD", func=summarize_job_description, description="Summarizes JD")]
jd_agent = initialize_agent(jd_tools, llm, agent="zero-shot-react-description", handle_parsing_errors=True)

def load_and_chunk_cv(pdf_input):
    st.info("Chunking the CV....")
    prefix = "Process CV:"
    if pdf_input.startswith(prefix):
        pdf_input = pdf_input[len(prefix):].strip()
    file_path = pdf_input.strip()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CV file not found at: {file_path}")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n\n".join([p.page_content for p in pages])
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    return {"chunks": chunks, "full_text": full_text}

cv_tools = [Tool(name="Process CV", func=load_and_chunk_cv, description="Loads and chunks CV PDF")]
cv_agent = initialize_agent(cv_tools, llm, agent="zero-shot-react-description", handle_parsing_errors=True)

def extract_name_email(cv_text):
    name_match = re.search(r"(?:Name[:\-]?\s*)([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)", cv_text)
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", cv_text)
    name = name_match.group(1).strip() if name_match else "Candidate"
    email = "ajaykumarmeeturi@gmail.com"  # Fixed email as per your configuration
    return name, email

def evaluate_eligibility(jd, cv, candidate_name, job_title):
    prompt = PromptTemplate(
        input_variables=["jd", "cv", "candidate_name", "job_title"],
        template="""
Given the following job summary:
{jd}

And the following candidate CV:
{cv}

Determine whether the candidate is suitable for the {job_title} position.
Return ONLY your answer in the EXACT following format (with no additional text):

Decision: <Suitable / Not Suitable>
Reason: <Justify the decision>
Basis: <Criteria used, e.g., skills, experience, education>
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.invoke({
        "jd": jd,
        "cv": "\n".join(cv),
        "candidate_name": candidate_name,
        "job_title": job_title
    })

eligibility_tools = [
    Tool(
        name="Evaluate Eligibility", 
        func=lambda x: evaluate_eligibility(**json.loads(x)), 
        description="JD-CV Matching"
    )
]
eligibility_agent = initialize_agent(eligibility_tools, llm, agent="zero-shot-react-description", handle_parsing_errors=True)

def send_email_agent(data):
    st.info("Processing the email content.....")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            print(f"Error parsing email data: {e}")
            data = {}
    candidate_name = data.get('name', 'Candidate')
    candidate_email = "ajaykumarmeeturi@gmail.com"  # Fixed email address
    is_selected = data.get('is_selected', False)

    sender_email = "meeturiajaykumar.23@gmail.com"
    sender_password = google_app_password

    if is_selected:
        subject = "🎉 You're Selected - Congratulations!"
        body = f"""
Hi {candidate_name},

We are pleased to inform you that you have been selected for the next steps in our hiring process!

Congratulations on making it this far. Our team was really impressed with your profile.

We will be sending more details shortly.

Best Regards,
The Team
        """
        message_text = "You have been selected."
    else:
        subject = "Update on Your Job Application"
        body = f"""
Hi {candidate_name},

Thank you for taking the time to apply and participate in our screening process.

After careful consideration, we regret to inform you that we won't be moving forward with your application at this time.

We appreciate your interest and encourage you to apply for future opportunities with us.

Best Wishes,
The Team
        """
        message_text = "You have not been selected."

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = candidate_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, candidate_email, message.as_string())
        server.quit()
        result = "✅ Email sent to the Candidate."
        print(result)
        return result
    except Exception as e:
        result = f"❌ Failed to send email to {candidate_email}: {e}"
        print(result)
        return result

email_tools = [Tool(name="Send Email", func=lambda x: send_email_agent(x), description="Sends result email")]
email_agent = initialize_agent(email_tools, llm, agent="zero-shot-react-description", handle_parsing_errors=True)

# --- Main UI ---
st.header("🧑‍💻 TalentSync AI 🤖")
st.markdown("###### Empowering Recruitment with Intelligent Multi-Agent Screening & Insightful Job-CV Matching 🧑‍💻")

# Use the uploaded JD and CV files if available; otherwise use the existing files.
CSV_PATH = JD_FILE_PATH  # Use the JD file path from the sidebar upload
# CV_FOLDER remains unchanged as defined above

df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
roles = ["Select role"] + df["Job Title"].tolist()
cv_files = ["Select CV"] + [f for f in os.listdir(CV_FOLDER) if f.endswith(".pdf")]

chosen_role = st.selectbox("Select a Job Title:", roles)

if chosen_role != "Select role":
    jd_text = df[df["Job Title"] == chosen_role]["Job Description"].values[0]
    if st.toggle("📄 View Job Description"):
        st.markdown("#### ✍️ Job Description")
        st.write(jd_text)

chosen_cv = st.selectbox("Select a CV:", cv_files)

if chosen_cv != "Select CV":
    if st.toggle("📄 View CV"):
        pdf_path = os.path.join(CV_FOLDER, chosen_cv)
        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>',
            unsafe_allow_html=True
        )

if chosen_role != "Select role" and chosen_cv != "Select CV":
    if st.button("🚀 Run Multi-Agent Screening"):
        pdf_path = os.path.join(CV_FOLDER, chosen_cv)
        with st.expander("Processing Steps", expanded=True):
            st.info("Loading CV..")
        
        jd_result = jd_agent.invoke({"input": f"Summarize JD: {jd_text}"})
        jd_summary = get_output(jd_result)
        embed_and_store([jd_summary], "jd")
        
        cv_result = cv_agent.invoke({"input": f"Process CV: {pdf_path}"})
        cv_data = get_output(cv_result)
        
        if isinstance(cv_data, dict):
            cv_chunks = cv_data.get("chunks", [])
            full_cv_text = cv_data.get("full_text", "")
        else:
            cv_chunks = [cv_data]
            full_cv_text = " ".join(cv_chunks)

        embed_and_store(cv_chunks, "cv")
        extracted_name, extracted_email = extract_name_email(full_cv_text)

        eligibility_input = {
            "jd": jd_summary,
            "cv": cv_chunks,
            "candidate_name": extracted_name,
            "job_title": chosen_role
        }
        eligibility_result_raw = eligibility_agent.invoke({"input": json.dumps(eligibility_input)})
        with st.expander("Processing Steps", expanded=True):
            st.info("Evaluating Eligibility and waiting for response...")
        eligibility_result = get_output(eligibility_result_raw)

        with st.expander("Results", expanded=True):
            st.markdown("### 🧠 Screening Result")
            st.success(eligibility_result)

        email_data = {
            "name": extracted_name,
            "email": extracted_email,
            "is_selected": "Suitable" in eligibility_result
        }
        email_result = email_agent.invoke({"input": email_data})
        with st.expander("Results", expanded=True):
            st.markdown("### 📧 Email Status")
            st.success(get_output(email_result))
