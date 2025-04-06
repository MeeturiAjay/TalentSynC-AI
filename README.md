# 🧑‍💻 TalentSync AI 🤖

TalentSync AI is a user-friendly recruitment tool that leverages smart automation to match job descriptions with candidate resumes. It streamlines the hiring process by comparing job requirements with candidate profiles, making it easier for recruiters to quickly identify the best-fit candidates.

---

## 🚀 Features

- **Job Description Management:** Easily add and manage job descriptions.
- **Resume Screening:** Upload candidate resumes in PDF format for automated screening.
- **Smart Matching:** Uses AI to analyze and match job requirements with candidate skills.
- **Visual Feedback:** Provides clear, real-time visual feedback during the screening process.
- **Automated Email Notifications:** Notifies candidates about their screening results via email.

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **AI & NLP:** [LangChain](https://www.langchain.com/) and ChatGroq
- **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter
- **Embeddings & Vector Store:** HuggingFaceEmbeddings, FAISS
- **Email Integration:** smtplib, MIME (email.mime)
- **Environment Management:** python-dotenv

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TalentSync-AI.git
cd TalentSync-AI
```

### 2. Set Up the Environment
Ensure you have Python 3.7 or above installed.

Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a .env file in the root directory and add:
```
GROQ_API_KEY=your_groq_api_key
GOOGLE_APP_PASSWORD=your_google_app_password
```
| 📝 Replace your_groq_api_key and your_google_app_password with your actual credentials.

### 5. Run the Application
```
streamlit run app.py
```
---

### 💡 How It Works
1. Enter Job Description in the sidebar.

2. Upload Candidate CV (PDF format) using the sidebar.

3. Click on "🚀 Run Multi-Agent Screening" to:

- Summarize the job description.

- Extract and analyze key information from the CV.

- Match CV data against the job profile.

- Notify the candidate about their result via email.

4. View Results: Check screening status and email delivery details on-screen.

---

### 🤝 Contributing
Contributions are welcome! Feel free to:

- Report bugs

- Open issues

- Suggest new features

- Submit pull requests

---

### 📬 Contact
For questions or suggestions, feel free to reach out via meeturiajaykumar.23@gmail.com

---

⭐ Don't forget to star this repo if you found it useful!
