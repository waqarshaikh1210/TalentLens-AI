# 🎯 TalentLens AI – Resume Screening System

TalentLens AI is an intelligent resume screening application that helps recruiters automatically rank candidates based on job description relevance using NLP techniques.

---

## 🚀 Features

- 📄 Upload multiple PDF resumes
- 🧠 NLP-based matching (TF-IDF + Cosine Similarity)
- 🎯 Skill-based matching system
- 🏆 Automatic candidate ranking
- 📊 Score visualization (bar chart)
- 📥 Export results as CSV
- 🌙 Modern dark UI (Streamlit)

---

## 🧠 How It Works

### 1. Text Extraction
- Extracts text from PDF resumes using PyPDF2

### 2. NLP Processing
- Converts text into numerical vectors using **TF-IDF**
- Measures similarity using **Cosine Similarity**

### 3. Skill Matching
- Matches predefined skills with:
  - Job Description
  - Resume content

### 4. Final Score
Final score is calculated as:
