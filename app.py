import streamlit as st
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Screening", layout="wide")

# Sidebar
st.sidebar.title("🤖 AI Resume Screening")
st.sidebar.write("Capstone Project Dashboard")

# Main Title
st.title("AI Resume Screening & Candidate Ranking System")

st.write("Upload resumes and compare them with the job description.")

# Job description input
job_description = st.text_area("📄 Enter Job Description")

# Upload resumes
uploaded_files = st.file_uploader(
    "📂 Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Extract text from PDF
def extract_text(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    except:
        text = ""
    return text

# Skills list
skills = [
    "python","machine learning","sql","data analysis",
    "java","html","css","javascript","react",
    "pandas","numpy","excel","power bi"
]

# Skill detection
def extract_skills(text):
    found = []
    text = text.lower()
    for skill in skills:
        if skill in text:
            found.append(skill)
    return found


if st.button("🔍 Analyze Resumes"):

    resumes = []
    names = []
    skill_data = []

    for file in uploaded_files:
        text = extract_text(file)
        resumes.append(text)
        names.append(file.name)

        detected = extract_skills(text)
        skill_data.append(", ".join(detected))

    # AI similarity
    documents = [job_description] + resumes

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )

    scores = similarity.flatten()

    results = pd.DataFrame({
        "Candidate": names,
        "Match Score": scores,
        "Skills": skill_data
    })

    results = results.sort_values(
        by="Match Score",
        ascending=False
    )

    # Resume count
    st.info(f"Total Resumes Uploaded: {len(uploaded_files)}")

    # Top candidate
    best = results.iloc[0]
    st.success(
        f"🏆 Top Candidate: {best['Candidate']} (Score: {best['Match Score']:.2f})"
    )

    # Ranking table
    st.subheader("📋 Candidate Ranking")
    st.dataframe(results)

    # BAR GRAPH
    st.subheader("📊 Candidate Score Bar Graph")
    st.bar_chart(results.set_index("Candidate")["Match Score"])

    # Download results
    csv = results.to_csv(index=False)

    st.download_button(
        "⬇ Download Ranking Results",
        csv,
        "ranking_results.csv"
    )