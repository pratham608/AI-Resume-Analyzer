import os

import streamlit as st
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ==========================================
# GEMINI CLIENT
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key nahi mili. Please VS Code restart karke dobara run karo.")
    st.stop()

client = genai.Client(
    api_key=api_key
)


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ==========================================
# TITLE
# ==========================================

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and compare it with a job description "
    "using NLP and Generative AI."
)


# ==========================================
# RESUME UPLOAD
# ==========================================

resume = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)


# ==========================================
# JOB DESCRIPTION
# ==========================================

job_description = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the job description here..."
)


# ==========================================
# RESUME PROCESSING
# ==========================================

if resume is not None:

    st.success("Resume uploaded successfully! ✅")

    reader = PdfReader(resume)

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text


    # ======================================
    # SHOW RESUME TEXT
    # ======================================

    st.subheader("📄 Extracted Resume Text")

    st.text_area(
        "Resume Content",
        resume_text,
        height=300
    )


    # ======================================
    # MATCHING
    # ======================================

    if job_description.strip():

        # ----------------------------------
        # TF-IDF
        # ----------------------------------

        vectorizer = TfidfVectorizer()

        vectors = vectorizer.fit_transform(
            [
                resume_text,
                job_description
            ]
        )


        # ----------------------------------
        # COSINE SIMILARITY
        # ----------------------------------

        similarity = cosine_similarity(
            vectors[0],
            vectors[1]
        )[0][0]


        match_percentage = similarity * 100


        # ==================================
        # MATCH SCORE
        # ==================================

        st.subheader("🎯 Resume Match Score")

        st.metric(
            "Job Match",
            f"{match_percentage:.2f}%"
        )


        # ==================================
        # SKILLS LIST
        # ==================================

        skills = [

            "Python",
            "Machine Learning",
            "Data Science",
            "SQL",
            "NLP",
            "Natural Language Processing",
            "Generative AI",
            "Git",
            "GitHub",
            "NumPy",
            "Pandas",
            "Scikit-learn",
            "TensorFlow",
            "PyTorch",
            "Deep Learning",
            "LLM",
            "API",
            "Cloud",
            "Data Analysis",
            "Data Visualization"

        ]


        resume_lower = resume_text.lower()

        job_lower = job_description.lower()


        matched_skills = []

        missing_skills = []


        # ==================================
        # SKILL CHECK
        # ==================================

        for skill in skills:

            if skill.lower() in job_lower:

                if skill.lower() in resume_lower:

                    matched_skills.append(skill)

                else:

                    missing_skills.append(skill)


        # ==================================
        # SKILLS ANALYSIS
        # ==================================

        st.subheader("🧠 Skills Analysis")


        # ----------------------------------
        # MATCHED SKILLS
        # ----------------------------------

        st.write("### ✅ Matched Skills")

        if matched_skills:

            st.success(
                ", ".join(matched_skills)
            )

        else:

            st.info(
                "No matched skills found."
            )


        # ----------------------------------
        # MISSING SKILLS
        # ----------------------------------

        st.write("### ❌ Missing Skills")

        if missing_skills:

            st.error(
                ", ".join(missing_skills)
            )

        else:

            st.success(
                "No major missing skills!"
            )


        # ==================================
        # IMPROVEMENT SUGGESTIONS
        # ==================================

        st.subheader("💡 Resume Improvement Suggestions")

        if missing_skills:

            st.write(
                "Consider adding the following skills "
                "if you genuinely have experience with them:"
            )

            for skill in missing_skills:

                st.write(
                    f"• {skill}"
                )

        else:

            st.success(
                "Your resume covers the major skills "
                "mentioned in the job description!"
            )


        # ==================================
        # GEMINI AI ANALYSIS
        # ==================================

        st.subheader("🤖 Gemini AI Resume Feedback")

        analyze_button = st.button(
            "🚀 Analyze Resume with Gemini"
        )


        if analyze_button:

            with st.spinner(
                "Gemini is analyzing your resume..."
            ):

                prompt = f"""
You are an expert technical recruiter and resume reviewer.

Analyze the candidate's resume against the provided job description.

Give the response in simple and practical language.

Include these sections:

1. Overall Resume Assessment
2. Strengths
3. Missing or Weak Skills
4. Resume Improvement Suggestions
5. Projects Suggestions
6. Interview Preparation Topics
7. Final Recommendation

Important:
- Do not invent experience.
- Do not invent skills.
- Only use information actually present in the resume.
- Give practical suggestions for improving the resume.

RESUME:

{resume_text}

JOB DESCRIPTION:

{job_description}

Current TF-IDF Match Score:

{match_percentage:.2f}%
"""


                try:

                    response = client.models.generate_content(

                        model="gemini-2.5-flash",

                        contents=prompt

                    )


                    st.markdown(
                        response.text
                    )


                except Exception as e:

                    st.error(
                        "Gemini API error occurred."
                    )

                    st.write(
                        str(e)
                    )