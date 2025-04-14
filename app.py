import os
import base64
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit UI Header
st.header("🧑‍💻 Multi-Agent Job Screening 🏢")

# Paths
CSV_PATH = os.path.join("Data", "job_description.csv")
CV_FOLDER = "Data/CVs1"

# Load Job Descriptions
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")
roles = ["Select role"] + df["Job Title"].tolist()

# Load CVs
cv_files = ["Select CV"] + [f for f in os.listdir(CV_FOLDER) if f.endswith(".pdf")]

# **Job Description Section**
chosen_role = st.selectbox("Select a Job Title:", roles)

if chosen_role != "Select role":
    show_description = st.toggle("View Description")

    if show_description:
        job_desc = df[df["Job Title"] == chosen_role]["Job Description"].values

        if len(job_desc) > 0:
            job_desc = job_desc[0]  # Extract text
        else:
            st.error("No job description available for this role.")
            st.stop()

        # Formatting Job Description
        sections = ["Job Description", "Responsibilities", "Qualifications"]
        formatted_desc = ""

        for i, section in enumerate(sections):
            if section in job_desc:
                split_text = job_desc.split(section + ":")
                section_content = split_text[1].strip() if len(split_text) > 1 else ""

                if i + 1 < len(sections) and sections[i + 1] in section_content:
                    section_content = section_content.split(sections[i + 1] + ":")[0]

                formatted_desc += f"### {section}\n"
                formatted_desc += "- " + section_content.replace(". ", ".\n- ") + "\n\n"

        st.markdown(formatted_desc)

# **CV Selection & Viewing**
chosen_CV = st.selectbox("Select a CV", options=cv_files)

if chosen_CV != "Select CV":
    show_cv = st.toggle("View CV")

    if show_cv:
        pdf_path = os.path.join(CV_FOLDER, chosen_CV)

        # Display PDF in iframe
        with open(pdf_path, "rb") as pdf_file:
            base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

        # Load and Chunk the PDF using LangChain
        cv_loader = PyPDFLoader(pdf_path)
        pages = cv_loader.load()
        text = "\n\n".join([page.page_content for page in pages])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(text)
        
