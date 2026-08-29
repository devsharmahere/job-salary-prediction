import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "salary_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "models" / "feature_columns.pkl"
SKILL_COLUMNS_PATH = BASE_DIR / "models" / "skill_columns.pkl"


# -----------------------------
# Load trained model
# -----------------------------

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
skill_columns = joblib.load(SKILL_COLUMNS_PATH)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Job Salary Predictor",
    page_icon="💰",
    layout="wide"
)


# -----------------------------
# Page title
# -----------------------------

st.markdown(
    """
    <h1 style="text-align: center;">💰 Job Salary Predictor</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align: center; margin-bottom: 80px;">
        Predict the annual salary of a professional based on job-related characteristics.
    </p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Job Information
# -----------------------------

st.header("Job Information")

col1, col2 = st.columns(2)

with col1:
    job_category = st.selectbox(
        "Job Category",
        [
            "AI Engineering",
            "Architecture",
            "Business",
            "Data Engineering",
            "Data Science",
            "Governance",
            "Infrastructure",
            "ML Operations",
            "Product",
            "Research",
            "Robotics",
            "Security"
        ]
    )

    experience_level = st.selectbox(
        "Experience Level",
        [
            "Entry (0-2 yrs)",
            "Lead (10+ yrs)",
            "Mid (3-5 yrs)",
            "Senior (6-9 yrs)"
        ]
    )

with col2:
    city = st.selectbox(
        "City",
        [
            "Amsterdam",
            "Austin",
            "Bangalore",
            "Beijing",
            "Berlin",
            "Boston",
            "Chicago",
            "Dubai",
            "London",
            "Los Angeles",
            "New York",
            "Paris",
            "Remote",
            "San Francisco",
            "Seattle",
            "Singapore",
            "Sydney",
            "Tokyo",
            "Toronto",
            "Zurich"
        ]
    )

    country = st.selectbox(
        "Country",
        [
            "Australia",
            "Canada",
            "China",
            "France",
            "Germany",
            "Global",
            "India",
            "Japan",
            "Netherlands",
            "Singapore",
            "Switzerland",
            "UAE",
            "UK",
            "USA"
        ]
    )

# -----------------------------
# Work Information
# -----------------------------

st.header("Work Information")

col1, col2 = st.columns(2)

with col1:
    years_of_experience = st.number_input(
        "Years of Experience",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    demand_score = st.number_input(
        "Demand Score",
        min_value=0.0,
        value=50.0,
        step=0.1
    )

with col2:
    demand_growth_yoy_pct = st.number_input(
        "Demand Growth YoY (%)",
        value=0.0,
        step=0.1
    )

    benefits_score_10 = st.number_input(
        "Benefits Score (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

# -----------------------------
# Remote Work
# -----------------------------

remote_work = st.selectbox(
    "Remote Work",
    [
        "Fully Remote",
        "Hybrid",
        "On-site"
    ]
)

# -----------------------------
# Education
# -----------------------------

education_required = st.selectbox(
    "Education Required",
    [
        "Associate's",
        "Bachelor's",
        "Bootcamp/Self-taught",
        "Master's",
        "PhD"
    ]
)

education_mapping = {
    "Associate's": 0,
    "Bachelor's": 1,
    "Bootcamp/Self-taught": 2,
    "Master's": 3,
    "PhD": 4
}

# -----------------------------
# Required Skills
# -----------------------------

st.header("Required Skills")

selected_skills = st.multiselect(
    "Select Required Skills",
    skill_columns
)
# -----------------------------
# Prepare Input Data
# -----------------------------

def prepare_input():
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    # Numerical features
    input_data["years_of_experience"] = years_of_experience
    input_data["demand_score"] = demand_score
    input_data["demand_growth_yoy_pct"] = demand_growth_yoy_pct
    input_data["benefits_score_10"] = benefits_score_10

    # One-hot encoded categorical features
    input_data[f"job_category_{job_category}"] = 1
    input_data[f"experience_level_{experience_level}"] = 1
    input_data[f"city_{city}"] = 1
    input_data[f"country_{country}"] = 1
    input_data[f"remote_work_{remote_work}"] = 1

    # Education
    education_code = education_mapping[education_required]
    input_data[f"education_required_{education_code}"] = 1

    # Skills
    for skill in selected_skills:
        input_data[skill] = 1

    return input_data

# -----------------------------
# Salary Prediction
# -----------------------------

if st.button("Predict Salary", type="primary"):

    input_data = prepare_input()

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Annual Salary: ${prediction:,.2f}"
    )