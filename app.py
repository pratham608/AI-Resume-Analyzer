import os
import re
import time

import streamlit as st
from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# GEMINI API SETUP
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:

    st.error(
        "❌ Gemini API key nahi mili.\n\n"
        "Please check GEMINI_API_KEY environment variable "
        "and restart VS Code."
    )

    st.stop()


try:

    client = genai.Client(
        api_key=api_key
    )

except Exception as e:

    st.error(
        "❌ Gemini client create nahi ho paya."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# PAGE TITLE
# ============================================================

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and compare it with a job description "
    "using NLP, ATS analysis and Generative AI."
)


# ============================================================
# RESUME UPLOAD
# ============================================================

resume = st.file_uploader(
    "📤 Upload your Resume (PDF)",
    type=["pdf"]
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "💼 Paste Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)


# ============================================================
# SKILLS DATABASE
# ============================================================

skills = [

    # Programming
    "Python",
    "Java",
    "C++",
    "C",
    "JavaScript",
    "TypeScript",

    # Data Analytics
    "SQL",
    "Excel",
    "Power BI",
    "Tableau",
    "Data Analysis",
    "Data Visualization",
    "Statistics",

    # AI / ML
    "Artificial Intelligence",
    "AI",
    "Machine Learning",
    "Deep Learning",
    "NLP",
    "Natural Language Processing",
    "Generative AI",
    "GenAI",
    "LLM",
    "Large Language Models",
    "Computer Vision",

    # Python Libraries
    "NumPy",
    "Pandas",
    "Matplotlib",
    "Seaborn",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",

    # Web Development
    "HTML",
    "CSS",
    "React",
    "Node.js",
    "API",
    "REST API",

    # Tools
    "Git",
    "GitHub",
    "Docker",
    "Linux",

    # Cloud
    "AWS",
    "Azure",
    "Google Cloud",
    "Cloud Computing",

    # Databases
    "MySQL",
    "PostgreSQL",
    "MongoDB",

    # Core CS
    "Data Structures",
    "Algorithms",
    "Problem Solving"
]


# ============================================================
# FUNCTION: CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# FUNCTION: FIND SKILLS
# ============================================================

def find_skills(text):

    text_lower = clean_text(text)

    found_skills = []

    for skill in skills:

        skill_lower = skill.lower()

        if skill_lower in text_lower:

            found_skills.append(skill)

    return found_skills


# ============================================================
# FUNCTION: EXTRACT KEYWORDS
# ============================================================

def extract_keywords(text):

    text = clean_text(text)

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9+#.-]{2,}\b",
        text
    )

    stop_words = {

        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "your",
        "you",
        "are",
        "will",
        "our",
        "have",
        "has",
        "job",
        "work",
        "role",
        "using",
        "use",
        "required",
        "requirements",
        "experience",
        "years",
        "candidate",
        "team",
        "good",
        "strong",
        "skills",
        "knowledge",
        "ability",
        "looking",
        "responsibilities",
        "including",
        "preferred",
        "must"
    }

    keywords = []

    for word in words:

        if word not in stop_words:

            if word not in keywords:

                keywords.append(word)

    return keywords


# ============================================================
# FUNCTION: GEMINI ANALYSIS WITH RETRY + FALLBACK
# ============================================================

def analyze_with_gemini(prompt):

    # Primary model + fallback models
    models_to_try = [

        "gemini-3.8-flash",

        "gemini-3.7-flash",

        "gemini-3.5-flash-lite"

    ]

    last_error = None

    # Try each model
    for model_name in models_to_try:

        # Retry same model 2 times
        for attempt in range(2):

            try:

                response = client.models.generate_content(

                    model=model_name,

                    contents=prompt

                )

                if response and response.text:

                    return response.text, model_name, None

            except Exception as e:

                last_error = e

                error_text = str(e).lower()

                # Temporary server errors
                temporary_error = (

                    "503" in error_text
                    or
                    "unavailable" in error_text
                    or
                    "high demand" in error_text
                    or
                    "429" in error_text
                    or
                    "resource exhausted" in error_text

                )

                if temporary_error:

                    # Wait before retry
                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                else:

                    # For non-temporary errors,
                    # move to next model
                    break


    return None, None, last_error


# ============================================================
# RESUME PROCESSING
# ============================================================

if resume is not None:

    st.success(
        "Resume uploaded successfully! ✅"
    )


    # ========================================================
    # READ PDF
    # ========================================================

    try:

        reader = PdfReader(resume)

        resume_text = ""

        for page in reader.pages:

            text = page.extract_text()

            if text:

                resume_text += text + "\n"


    except Exception as e:

        st.error(
            "❌ Resume PDF read nahi ho payi."
        )

        st.code(str(e))

        st.stop()


    # ========================================================
    # EMPTY PDF CHECK
    # ========================================================

    if not resume_text.strip():

        st.error(
            "❌ Resume se text extract nahi ho paya.\n\n"
            "Please ek text-based PDF upload karo."
        )

        st.stop()


    # ========================================================
    # EXTRACTED RESUME TEXT
    # ========================================================

    with st.expander(
        "📄 View Extracted Resume Text"
    ):

        st.text_area(
            "Resume Content",
            resume_text,
            height=300
        )


    # ========================================================
    # JOB DESCRIPTION CHECK
    # ========================================================

    if job_description.strip():

        # ====================================================
        # CLEAN TEXT
        # ====================================================

        resume_clean = clean_text(
            resume_text
        )

        job_clean = clean_text(
            job_description
        )


        # ====================================================
        # TF-IDF MATCHING
        # ====================================================

        try:

            vectorizer = TfidfVectorizer(
                stop_words="english"
            )

            vectors = vectorizer.fit_transform(
                [
                    resume_clean,
                    job_clean
                ]
            )

            similarity = cosine_similarity(
                vectors[0],
                vectors[1]
            )[0][0]

            match_percentage = similarity * 100

        except Exception:

            match_percentage = 0


        # ====================================================
        # FIND SKILLS
        # ====================================================

        resume_skills = find_skills(
            resume_text
        )

        job_skills = find_skills(
            job_description
        )


        resume_skill_set = {
            skill.lower()
            for skill in resume_skills
        }


        matched_skills = []

        missing_skills = []


        for skill in job_skills:

            if skill.lower() in resume_skill_set:

                matched_skills.append(
                    skill
                )

            else:

                missing_skills.append(
                    skill
                )


        # ====================================================
        # SKILL MATCH %
        # ====================================================

        if len(job_skills) > 0:

            skill_match_percentage = (

                len(matched_skills)
                /
                len(job_skills)

            ) * 100

        else:

            skill_match_percentage = 0


        # ====================================================
        # KEYWORD ANALYSIS
        # ====================================================

        job_keywords = extract_keywords(
            job_description
        )

        resume_keywords = extract_keywords(
            resume_text
        )


        resume_keyword_set = set(
            resume_keywords
        )


        matched_keywords = []

        missing_keywords = []


        for keyword in job_keywords:

            if keyword in resume_keyword_set:

                matched_keywords.append(
                    keyword
                )

            else:

                missing_keywords.append(
                    keyword
                )


        # ====================================================
        # KEYWORD MATCH %
        # ====================================================

        if len(job_keywords) > 0:

            keyword_match_percentage = (

                len(matched_keywords)
                /
                len(job_keywords)

            ) * 100

        else:

            keyword_match_percentage = 0


        # ====================================================
        # ATS SCORE
        # ====================================================

        ats_score = (

            (match_percentage * 0.40)
            +
            (skill_match_percentage * 0.40)
            +
            (keyword_match_percentage * 0.20)

        )

        ats_score = min(
            ats_score,
            100
        )


        # ====================================================
        # DASHBOARD
        # ====================================================

        st.divider()

        st.header(
            "📊 Resume Analysis Dashboard"
        )


        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "🎯 Job Match",
                f"{match_percentage:.1f}%"
            )


        with col2:

            st.metric(
                "🤖 ATS Score",
                f"{ats_score:.1f}%"
            )


        with col3:

            st.metric(
                "✅ Matched Skills",
                len(matched_skills)
            )


        with col4:

            st.metric(
                "❌ Missing Skills",
                len(missing_skills)
            )


        # ====================================================
        # ATS PROGRESS
        # ====================================================

        st.subheader(
            "🤖 ATS Compatibility"
        )

        st.progress(
            int(ats_score)
        )


        if ats_score >= 80:

            st.success(
                "Strong ATS compatibility based on this project's analysis."
            )

        elif ats_score >= 60:

            st.warning(
                "Moderate ATS compatibility. "
                "Some improvements may help."
            )

        else:

            st.error(
                "Low ATS compatibility. "
                "Resume alignment can be improved."
            )


        # ====================================================
        # SKILLS ANALYSIS
        # ====================================================

        st.divider()

        st.header(
            "🧠 Skills Analysis"
        )


        col1, col2 = st.columns(2)


        # ====================================================
        # MATCHED SKILLS
        # ====================================================

        with col1:

            st.subheader(
                "✅ Matched Skills"
            )

            if matched_skills:

                st.success(
                    ", ".join(
                        matched_skills
                    )
                )

            else:

                st.info(
                    "No matched skills found."
                )


        # ====================================================
        # MISSING SKILLS
        # ====================================================

        with col2:

            st.subheader(
                "❌ Missing Skills"
            )

            if missing_skills:

                st.error(
                    ", ".join(
                        missing_skills
                    )
                )

            else:

                st.success(
                    "No major missing skills found!"
                )


        # ====================================================
        # KEYWORD ANALYSIS
        # ====================================================

        st.divider()

        st.header(
            "🔑 ATS Keyword Analysis"
        )


        col1, col2 = st.columns(2)


        # ====================================================
        # MATCHED KEYWORDS
        # ====================================================

        with col1:

            st.subheader(
                "✅ Matched Keywords"
            )

            if matched_keywords:

                st.write(
                    ", ".join(
                        matched_keywords[:30]
                    )
                )

            else:

                st.info(
                    "No matching keywords found."
                )


        # ====================================================
        # MISSING KEYWORDS
        # ====================================================

        with col2:

            st.subheader(
                "⚠️ Missing Keywords"
            )

            if missing_keywords:

                st.write(
                    ", ".join(
                        missing_keywords[:30]
                    )
                )

            else:

                st.success(
                    "No major missing keywords."
                )


        # ====================================================
        # RESUME IMPROVEMENT
        # ====================================================

        st.divider()

        st.header(
            "💡 Resume Improvement Suggestions"
        )


        if missing_skills:

            st.write(
                "Consider adding these skills "
                "only if you genuinely have experience with them:"
            )

            for skill in missing_skills:

                st.write(
                    f"• {skill}"
                )

        else:

            st.success(
                "Your resume covers the major skills "
                "found in the job description."
            )


        if missing_keywords:

            st.write(
                "### 🔑 Important Missing Keywords"
            )

            st.write(
                ", ".join(
                    missing_keywords[:20]
                )
            )


        # ====================================================
        # GEMINI AI ANALYSIS
        # ====================================================

        st.divider()

        st.header(
            "🤖 Gemini AI Resume Feedback"
        )


        st.caption(
            "Gemini AI provides additional resume feedback. "
            "ATS calculations above work independently."
        )


        analyze_button = st.button(
            "🚀 Analyze Resume with Gemini",
            type="primary"
        )


        if analyze_button:

            # =================================================
            # GEMINI PROMPT
            # =================================================

            prompt = f"""

You are an expert technical recruiter,
ATS resume reviewer and career advisor.

Analyze the candidate's resume against the provided job description.

Use simple and practical language.

IMPORTANT RULES:

1. Do not invent experience.
2. Do not invent projects.
3. Do not invent skills.
4. Only use information actually present in the resume.
5. Clearly separate existing skills from recommended skills.
6. Give practical and actionable suggestions.
7. Do not guarantee a job or interview.
8. Do not exaggerate the candidate's experience.

Provide the following sections:

# 1. Overall Resume Assessment

Explain how well the resume matches the job description.

# 2. ATS Analysis

Explain:

- keyword alignment
- skill alignment
- job relevance
- possible content improvements

# 3. Strengths

List the strongest relevant points already present
in the resume.

# 4. Missing or Weak Skills

Mention important skills from the job description
that are not clearly present in the resume.

# 5. Missing Keywords

Mention important job-description keywords
that are absent or weakly represented.

# 6. Resume Improvement Suggestions

Give specific suggestions for improving the resume.

# 7. Project Suggestions

Suggest useful project ideas based on the job description.

Do NOT claim that the candidate already built them.

# 8. Interview Preparation

Give important technical topics to prepare.

# 9. Final Summary

Give a short practical summary of the main improvements.

----------------------------------------

RESUME:

{resume_text}

----------------------------------------

JOB DESCRIPTION:

{job_description}

----------------------------------------

PROJECT CALCULATED SCORES:

TF-IDF Job Match:
{match_percentage:.2f}%

Skill Match:
{skill_match_percentage:.2f}%

Keyword Match:
{keyword_match_percentage:.2f}%

ATS Score:
{ats_score:.2f}%

----------------------------------------

MATCHED SKILLS:

{", ".join(matched_skills)}

----------------------------------------

MISSING SKILLS:

{", ".join(missing_skills)}

----------------------------------------

MATCHED KEYWORDS:

{", ".join(matched_keywords[:30])}

----------------------------------------

MISSING KEYWORDS:

{", ".join(missing_keywords[:30])}

"""


            # =================================================
            # GEMINI CALL
            # =================================================

            with st.spinner(
                "🤖 Gemini is analyzing your resume..."
            ):

                result, model_used, error = (
                    analyze_with_gemini(
                        prompt
                    )
                )


            # =================================================
            # SUCCESS
            # =================================================

            if result:

                st.success(
                    f"Gemini analysis completed using {model_used} ✅"
                )

                st.markdown(
                    result
                )


            # =================================================
            # ERROR
            # =================================================

            else:

                st.error(
                    "❌ Gemini temporarily unavailable."
                )

                st.warning(
                    "Your ATS, skills and keyword analysis "
                    "is still available above."
                )

                if error:

                    st.code(
                        str(error)
                    )


    else:

        st.info(
            "💼 Please paste a Job Description "
            "to start the resume analysis."
        )