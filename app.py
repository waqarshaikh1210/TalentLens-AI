import streamlit as st
import PyPDF2
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(page_title="TalentLens AI", page_icon="🎯", layout="wide")

# ─────────────────────────────
# CUSTOM CSS
# ─────────────────────────────
st.markdown("""
<style>

/* ── Base — kill every white surface Streamlit injects ── */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stSidebar"],
[data-testid="stMain"],
[data-testid="block-container"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.main,
.main > div,
section[data-testid="stMain"],
div[data-testid="stMainBlockContainer"] {
    background-color: #0f172a !important;
    color: #f1f5f9;
}

/* ── Remove default top padding / white header bar ── */
[data-testid="stHeader"] {
    background: #0f172a !important;
    border-bottom: none !important;
}

/* ── Column blocks ── */
[data-testid="column"],
[data-testid="stColumn"] {
    background-color: #0f172a !important;
}

/* ── Any remaining white cards Streamlit wraps content in ── */
div[class*="stBlock"],
div[class*="element-container"],
div[class*="stElementContainer"],
section.main > div {
    background-color: #0f172a !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

/* ── Body text ── */
p, li, span, label, div {
    color: #cbd5e1;
}

/* ── Text area ── */
.stTextArea textarea {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
.stTextArea textarea::placeholder {
    color: #64748b !important;
}
.stTextArea textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.2) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #1e293b !important;
    border: 1.5px dashed #475569 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}
[data-testid="stFileUploader"] * {
    color: #94a3b8 !important;
}

/* ── Labels above inputs ── */
.stTextArea label, .stFileUploader label {
    color: #94a3b8 !important;
    font-size: 13px !important;
}

/* ── Analyze button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    height: 50px !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* ── Spinner text ── */
.stSpinner > div > span {
    color: #94a3b8 !important;
}

/* ── Alerts ── */
.stSuccess, .stWarning, .stError {
    border-radius: 8px !important;
}

/* ── Candidate card ── */
.candidate-card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.candidate-card.top {
    border: 1.5px solid #38bdf8;
    background-color: #0c2d48;
}
.candidate-rank {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
    margin-bottom: 2px;
}
.candidate-name {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 8px;
}
.candidate-score-label {
    font-size: 13px;
    color: #94a3b8;
    font-weight: 500;
}
.candidate-score-value {
    font-size: 28px;
    font-weight: 700;
    color: #38bdf8;
}
.score-bar-bg {
    background-color: #334155;
    border-radius: 6px;
    height: 6px;
    width: 100%;
    margin: 6px 0 14px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
}
.skills-label {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin-bottom: 8px;
}
.skill-tag {
    display: inline-block;
    background-color: #0f3460;
    color: #7dd3fc;
    border: 1px solid #1d4ed8;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px 4px 3px 0;
}
.top-badge {
    display: inline-block;
    background-color: #0ea5e9;
    color: #ffffff;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 10px;
    vertical-align: middle;
    letter-spacing: 0.5px;
}

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 30px 0;
}

/* ── Section heading ── */
.section-heading {
    font-size: 22px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 24px 0 16px;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# HEADER
# ─────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 10px;">
    <div style="font-size:42px; margin-bottom:6px;">🎯</div>
    <h1 style="font-size:36px; font-weight:800; color:#f8fafc; margin:0;">TalentLens AI</h1>
    <p style="font-size:16px; color:#64748b; margin-top:6px;">Upload resumes, enter a job description, and instantly rank your candidates.</p>
</div>
<hr class="section-divider"/>
""", unsafe_allow_html=True)

# ─────────────────────────────
# INPUT SECTION
# ─────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p style="font-size:16px; font-weight:600; color:#e2e8f0; margin-bottom:6px;">📌 Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        label="Job Description",
        label_visibility="hidden",
        height=230,
        placeholder="Paste the job description here.\n\nExample: Looking for a Python developer with experience in ML, NLP, SQL, and data analysis..."
    )

with col2:
    st.markdown('<p style="font-size:16px; font-weight:600; color:#e2e8f0; margin-bottom:6px;">📂 Upload Resumes (PDF)</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        label="Upload PDFs",
        label_visibility="hidden",
        type="pdf",
        accept_multiple_files=True
    )
    if uploaded_files:
        st.markdown(f'<p style="color:#38bdf8; font-size:13px; margin-top:8px;">✓ {len(uploaded_files)} file(s) ready</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────
# SKILLS DB
# ─────────────────────────────
SKILLS_DB = [
    "python", "machine learning", "deep learning", "nlp",
    "sql", "excel", "pandas", "numpy", "flask",
    "django", "tensorflow", "pytorch", "data analysis",
    "statistics", "communication"
]

# ─────────────────────────────
# FUNCTIONS
# ─────────────────────────────
def extract_text(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t.lower()
    except Exception:
        return ""
    return text

def extract_skills(text):
    return [s for s in SKILLS_DB if s in text]

def skill_match(jd, resume_skills):
    jd_lower = jd.lower()
    jd_skills = [s for s in SKILLS_DB if s in jd_lower]
    if not jd_skills:
        return 0, []
    matched = list(set(jd_skills).intersection(set(resume_skills)))
    score = len(matched) / len(jd_skills)
    return score, matched

# ─────────────────────────────
# ANALYZE BUTTON
# ─────────────────────────────
if st.button("⚡  Analyze Candidates", use_container_width=True):

    if not job_description.strip():
        st.warning("⚠️  Please enter a job description before analyzing.")
        st.stop()

    if not uploaded_files:
        st.warning("⚠️  Please upload at least one resume PDF.")
        st.stop()

    results = []

    with st.spinner("🔍  Analyzing resumes — this may take a moment..."):
        for file in uploaded_files:
            text = extract_text(file)
            if not text.strip():
                continue

            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([job_description, text])
            sim_score = cosine_similarity(tfidf[0:1], tfidf[1:])[0][0]

            resume_skills = extract_skills(text)
            skill_score, matched_skills = skill_match(job_description, resume_skills)

            final_score = (0.6 * sim_score) + (0.4 * skill_score)

            results.append({
                "name": file.name.replace(".pdf", ""),
                "score": final_score,
                "matched": matched_skills
            })

    if not results:
        st.error("❌  No readable text found in the uploaded resumes. Please check your files.")
        st.stop()

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    st.success(f"✅  Analysis complete! Ranked {len(results)} candidate(s).")

    # ─────────────────────────────
    # RANKED CARDS
    # ─────────────────────────────
    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">🏆 Ranked Candidates</div>', unsafe_allow_html=True)

    for i, r in enumerate(results):
        score_pct = int(r["score"] * 100)
        is_top = i == 0
        card_class = "candidate-card top" if is_top else "candidate-card"
        badge = '<span class="top-badge">BEST MATCH</span>' if is_top else ""

        if r["matched"]:
            skills_html = "".join([f'<span class="skill-tag">{s}</span>' for s in r["matched"]])
        else:
            skills_html = '<span style="color:#475569; font-size:13px;">No matching skills detected</span>'

        st.markdown(f"""
        <div class="{card_class}">
            <div class="candidate-rank">Rank #{i+1}</div>
            <div class="candidate-name">{r['name']}{badge}</div>
            <div class="candidate-score-label">Match Score</div>
            <div class="candidate-score-value">{score_pct}%</div>
            <div class="score-bar-bg">
                <div class="score-bar-fill" style="width:{score_pct}%;"></div>
            </div>
            <div class="skills-label">Matched Skills</div>
            <div>{skills_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────
    # CHART
    # ─────────────────────────────
    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">📊 Score Comparison</div>', unsafe_allow_html=True)

    names = [r["name"] for r in results]
    scores = [int(r["score"] * 100) for r in results]
    colors = ["#0ea5e9" if i == 0 else "#334155" for i in range(len(results))]

    fig, ax = plt.subplots(figsize=(9, max(2.5, len(results) * 0.75)))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")

    bars = ax.barh(names, scores, color=colors, height=0.55, edgecolor="none")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{score}%",
            va="center",
            ha="left",
            color="#94a3b8",
            fontsize=11
        )

    ax.set_xlabel("Match Score (%)", color="#64748b", fontsize=11)
    ax.set_xlim(0, max(scores) + 12)
    ax.tick_params(axis="y", colors="#e2e8f0", labelsize=11)
    ax.tick_params(axis="x", colors="#475569", labelsize=10)
    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    ax.xaxis.grid(True, color="#1e293b", linestyle="--", linewidth=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    st.pyplot(fig)

    # ─────────────────────────────
    # DOWNLOAD
    # ─────────────────────────────
    st.markdown('<hr class="section-divider"/>', unsafe_allow_html=True)

    df = pd.DataFrame([{
        "Rank": i + 1,
        "Candidate": r["name"],
        "Score (%)": int(r["score"] * 100),
        "Matched Skills": ", ".join(r["matched"]) if r["matched"] else "None"
    } for i, r in enumerate(results)])

    col_dl, col_empty = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="⬇  Download Results as CSV",
            data=df.to_csv(index=False),
            file_name="talentlens_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ─────────────────────────────
    # SUMMARY FOOTER
    # ─────────────────────────────
    top = results[0]
    st.markdown(f"""
    <div style="background-color:#0c2d48; border:1.5px solid #0ea5e9; border-radius:12px; padding:18px 24px; margin-top:16px;">
        <p style="margin:0; font-size:15px; color:#7dd3fc; font-weight:600;">
            🎯 Best Candidate: <span style="color:#f1f5f9;">{top['name']}</span>
            &nbsp;—&nbsp;
            <span style="color:#38bdf8;">{int(top['score']*100)}% match</span>
        </p>
    </div>
    """, unsafe_allow_html=True)