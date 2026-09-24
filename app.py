import streamlit as st
import pandas as pd
import joblib
import re
from pypdf import PdfReader


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmokeSense",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("smoking_status_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #f5f8fc;
    }

    /* Remove top spacing */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #173f6b, #197c91);
        padding: 38px 45px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(23,63,107,0.15);
    }

    .main-header h1 {
        font-size: 38px;
        margin: 0;
        font-weight: 700;
    }

    .main-header p {
        font-size: 16px;
        margin-top: 10px;
        color: #e5f4fa;
    }

    /* Step navigation */
    .steps {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 25px 0;
    }

    .step {
        padding: 10px 20px;
        border-radius: 25px;
        background: #e4eaf2;
        color: #53657b;
        font-size: 14px;
        font-weight: 600;
    }

    .step.active {
        background: #197c91;
        color: white;
    }

    /* Cards */
    .card {
        background: white;
        padding: 28px;
        border-radius: 16px;
        margin: 18px 0;
        box-shadow: 0 5px 20px rgba(30,55,80,0.07);
        border: 1px solid #e6ebf1;
    }

    .card-title {
        color: #173f6b;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .card-subtitle {
        color: #728198;
        font-size: 14px;
        margin-bottom: 20px;
    }

    /* Source labels */
    .report-source {
        background: #e7f8f1;
        border: 1px solid #a9e5ce;
        color: #18794e;
        padding: 7px 12px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px 0 10px 0;
    }

    .manual-source {
        background: #fff7df;
        border: 1px solid #f2d38a;
        color: #986b00;
        padding: 7px 12px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        margin: 3px 0 10px 0;
    }

    /* Information boxes */
    .info-box {
        background: #edf7ff;
        border-left: 4px solid #197c91;
        padding: 15px 18px;
        border-radius: 8px;
        color: #38526b;
        margin: 15px 0;
    }

    /* Success prediction */
    .prediction-card {
        background: #ffffff;
        border-radius: 22px;
        padding: 45px;
        text-align: center;
        border: 1px solid #dfe7ef;
        box-shadow: 0 10px 35px rgba(30,55,80,0.10);
        margin-top: 25px;
    }

    .prediction-title {
        color: #718096;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .prediction-value {
        font-size: 48px;
        font-weight: 800;
        margin: 5px 0 15px 0;
    }

    .smoker {
        color: #c0392b;
    }

    .non-smoker {
        color: #18865b;
    }

    .prediction-description {
        color: #68788d;
        font-size: 15px;
        max-width: 600px;
        margin: auto;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 9px;
        min-height: 45px;
        font-weight: 600;
        border: none;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8b98a9;
        font-size: 12px;
        margin-top: 50px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>SmokeSense</h1>
    <p>Health report analysis and machine-learning based smoking status prediction</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STEP FUNCTION
# ============================================================

def show_steps(current_step):

    names = [
        "01  Upload Reports",
        "02  Review Information",
        "03  Prediction"
    ]

    html = '<div class="steps">'

    for i, name in enumerate(names, start=1):

        if i == current_step:
            html += f'<div class="step active">{name}</div>'
        else:
            html += f'<div class="step">{name}</div>'

    html += '</div>'

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    try:

        reader = PdfReader(uploaded_file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += "\n" + page_text

        return text

    except Exception as e:

        st.error(f"Could not read {uploaded_file.name}: {e}")

        return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.replace("\xa0", " ")

    text = re.sub(r"[ \t]+", " ", text)

    return text


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def find_number(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            try:
                return float(match.group(1))
            except:
                continue

    return None


# ============================================================
# YES / NO EXTRACTION
# ============================================================

def find_yes_no(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip().lower()

            if value in ["yes", "y", "positive", "present"]:
                return "Yes"

            if value in ["no", "n", "negative", "absent"]:
                return "No"

    return None


# ============================================================
# HEARING EXTRACTION (returns "Normal" / "Abnormal")
# ============================================================

def find_hearing(text, patterns):

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).strip().lower()

            if value in ["normal", "no", "n", "negative"]:
                return "Normal"

            if value in ["abnormal", "yes", "y", "positive"]:
                return "Abnormal"

    return None


# ============================================================
# GENDER EXTRACTION
# ============================================================

def find_gender(text):

    patterns = [
        r"\bgender\s*[:\-]?\s*(male|female)",
        r"\bsex\s*[:\-]?\s*(male|female)",
        r"\b(male|female)\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            value = match.group(1).lower()

            if value == "male":
                return "Male"

            if value == "female":
                return "Female"

    return None


# ============================================================
# AGE EXTRACTION
# ============================================================

def find_age(text):

    patterns = [
        r"\bage\s*[:\-]?\s*(\d{1,3})\s*(?:years?|yrs?)?",
        r"\b(\d{1,3})\s*(?:years?|yrs?)\s*old\b"
    ]

    return find_number(text, patterns)


# ============================================================
# BLOOD PRESSURE
# ============================================================

def find_blood_pressure(text):

    # Combined "120/80" style patterns first

    combined_patterns = [
        r"blood\s*pressure\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        r"bp\s*[:\-]?\s*(\d{2,3})\s*/\s*(\d{2,3})"
    ]

    for pattern in combined_patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:

            return (
                float(match.group(1)),
                float(match.group(2))
            )

    # Separate systolic / diastolic lines (e.g. lab report tables)

    systolic = find_number(
        text,
        [
            r"systolic\s*(?:blood\s*pressure)?\s*[:\-]?\s*(\d{2,3})"
        ]
    )

    diastolic = find_number(
        text,
        [
            r"diastolic\s*(?:blood\s*pressure)?\s*[:\-]?\s*(\d{2,3})",
            r"relaxation\s*[:\-]?\s*(\d{2,3})"
        ]
    )

    return systolic, diastolic


# ============================================================
# EXTRACT ALL HEALTH VALUES
# ============================================================

def extract_health_data(text):

    text = clean_text(text)

    data = {}

    # Personal information

    data["age"] = find_age(text)

    data["gender"] = find_gender(text)

    data["height"] = find_number(
        text,
        [
            r"height\s*(?:\(cm\))?\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"height\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["weight"] = find_number(
        text,
        [
            r"weight\s*(?:\(kg\))?\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"weight\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["waist"] = find_number(
        text,
        [
            r"waist\s*(?:circumference)?\s*(?:\(cm\))?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    # Eyesight

    data["eyesight_left"] = find_number(
        text,
        [
            r"eyesight\s*\(?(?:left)\)?\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"left\s*eye\s*(?:vision|eyesight)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["eyesight_right"] = find_number(
        text,
        [
            r"eyesight\s*\(?(?:right)\)?\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"right\s*eye\s*(?:vision|eyesight)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    # Hearing

    data["hearing_left"] = find_hearing(
        text,
        [
            r"hearing\s*\(?(?:left)\)?\s*[:\-]?\s*(normal|abnormal|yes|no)",
            r"left\s*ear\s*(?:hearing)?\s*[:\-]?\s*(normal|abnormal|yes|no)"
        ]
    )

    data["hearing_right"] = find_hearing(
        text,
        [
            r"hearing\s*\(?(?:right)\)?\s*[:\-]?\s*(normal|abnormal|yes|no)",
            r"right\s*ear\s*(?:hearing)?\s*[:\-]?\s*(normal|abnormal|yes|no)"
        ]
    )

    # Blood pressure

    systolic, diastolic = find_blood_pressure(text)

    data["systolic"] = systolic
    data["relaxation"] = diastolic

    # Blood test values

    data["fasting_blood_sugar"] = find_number(
        text,
        [
            r"fasting\s*blood\s*sugar\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"fasting\s*glucose\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["cholesterol"] = find_number(
        text,
        [
            r"total\s*cholesterol\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"\bcholesterol\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["triglyceride"] = find_number(
        text,
        [
            r"triglyceride[s]?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["HDL"] = find_number(
        text,
        [
            r"\bHDL\s*(?:cholesterol)?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["LDL"] = find_number(
        text,
        [
            r"\bLDL\s*(?:cholesterol)?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["hemoglobin"] = find_number(
        text,
        [
            r"hemoglobin\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"\bHB\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["urine_protein"] = find_number(
        text,
        [
            r"urine\s*protein\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["serum_creatinine"] = find_number(
        text,
        [
            r"serum\s*creatinine\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"creatinine\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["AST"] = find_number(
        text,
        [
            r"\bAST\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"AST\s*\(?(?:SGOT)\)?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["ALT"] = find_number(
        text,
        [
            r"\bALT\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"ALT\s*\(?(?:SGPT)\)?\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    data["Gtp"] = find_number(
        text,
        [
            r"\bGTP\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"\bGGT\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    )

    # Dental

    data["dental_caries"] = find_yes_no(
        text,
        [
            r"dental\s*caries\s*[:\-]?\s*(yes|no|positive|negative|present|absent)",
            r"caries\s*[:\-]?\s*(yes|no|positive|negative|present|absent)"
        ]
    )

    data["tartar"] = find_yes_no(
        text,
        [
            r"tartar\s*[:\-]?\s*(yes|no|positive|negative|present|absent)",
            r"dental\s*tartar\s*[:\-]?\s*(yes|no|positive|negative|present|absent)"
        ]
    )

    return data


# ============================================================
# SESSION STATE
# ============================================================

if "step" not in st.session_state:
    st.session_state.step = 1

if "extracted" not in st.session_state:
    st.session_state.extracted = {}

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "uploaded_names" not in st.session_state:
    st.session_state.uploaded_names = []


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

if st.session_state.step == 1:

    show_steps(1)

    st.markdown("""
    <div class="card">
        <div class="card-title">Upload Health Reports</div>
        <div class="card-subtitle">
            Upload one or more PDF health reports. SmokeSense will extract
            the health values that are available in the reports.
        </div>
        <div class="info-box">
            You can upload multiple reports. For example, a blood test,
            health check-up report or blood pressure report.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Select your health reports",
        type=["pdf"],
        accept_multiple_files=True,
        key="report_uploader"
    )

    if uploaded_files:

        if st.button(
            "Analyze Reports",
            use_container_width=True,
            key="analyze_reports_button"
        ):

            combined_text = ""

            names = []

            with st.spinner("Reading your reports..."):

                for file in uploaded_files:

                    names.append(file.name)

                    text = extract_pdf_text(file)

                    combined_text += "\n" + text

            extracted = extract_health_data(combined_text)

            st.session_state.report_text = combined_text
            st.session_state.extracted = extracted
            st.session_state.uploaded_names = names

            st.session_state.step = 2

            st.rerun()


# ============================================================
# STEP 2 — REVIEW
# ============================================================

elif st.session_state.step == 2:

    show_steps(2)

    st.markdown("""
    <div class="card">
        <div class="card-title">Review Your Information</div>
        <div class="card-subtitle">
            Green fields were found in your uploaded reports.
            Yellow fields need to be provided manually.
        </div>
        <div class="report-source">
            GREEN — Value extracted from uploaded report
        </div>
        <div class="manual-source">
            YELLOW — Information not found in report
        </div>
    </div>
    """, unsafe_allow_html=True)

    extracted = st.session_state.extracted

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def source_label(value):

        if value is not None:

            st.markdown(
                '<div class="report-source">Taken from uploaded report — Please verify</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="manual-source">Not found in report — Please enter</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # PERSONAL INFORMATION
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Personal Information</div>
        <div class="card-subtitle">
            Basic information required by the prediction model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**Age (years)**")

        source_label(extracted.get("age"))

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=int(extracted["age"]) if extracted.get("age") is not None else 30,
            step=1,
            key="review_age",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Height (cm)**")

        source_label(extracted.get("height"))

        height = st.number_input(
            "Height",
            min_value=50.0,
            max_value=250.0,
            value=float(extracted["height"]) if extracted.get("height") is not None else 165.0,
            step=0.5,
            key="review_height",
            label_visibility="collapsed"
        )

    with c3:

        st.markdown("**Weight (kg)**")

        source_label(extracted.get("weight"))

        weight = st.number_input(
            "Weight",
            min_value=10.0,
            max_value=300.0,
            value=float(extracted["weight"]) if extracted.get("weight") is not None else 60.0,
            step=0.5,
            key="review_weight",
            label_visibility="collapsed"
        )

    c4, c5 = st.columns(2)

    with c4:

        st.markdown("**Waist circumference (cm)**")

        source_label(extracted.get("waist"))

        waist = st.number_input(
            "Waist",
            min_value=20.0,
            max_value=200.0,
            value=float(extracted["waist"]) if extracted.get("waist") is not None else 75.0,
            step=0.5,
            key="review_waist",
            label_visibility="collapsed"
        )

    with c5:

        st.markdown("**Gender**")

        source_label(extracted.get("gender"))

        gender_options = ["Female", "Male"]

        default_gender = (
            extracted["gender"]
            if extracted.get("gender") in gender_options
            else "Female"
        )

        gender = st.selectbox(
            "Gender",
            gender_options,
            index=gender_options.index(default_gender),
            key="review_gender",
            label_visibility="collapsed"
        )


    # ========================================================
    # VISION
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Vision</div>
        <div class="card-subtitle">
            Enter the eyesight values if they are available in your report.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("**Left eye eyesight**")

        source_label(extracted.get("eyesight_left"))

        eyesight_left = st.number_input(
            "Left Eye",
            min_value=0.0,
            max_value=10.0,
            value=float(extracted["eyesight_left"])
            if extracted.get("eyesight_left") is not None
            else 1.0,
            step=0.1,
            key="review_eyesight_left",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Right eye eyesight**")

        source_label(extracted.get("eyesight_right"))

        eyesight_right = st.number_input(
            "Right Eye",
            min_value=0.0,
            max_value=10.0,
            value=float(extracted["eyesight_right"])
            if extracted.get("eyesight_right") is not None
            else 1.0,
            step=0.1,
            key="review_eyesight_right",
            label_visibility="collapsed"
        )


    # ========================================================
    # HEARING
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Hearing</div>
        <div class="card-subtitle">
            Select the hearing status if it was not included in the report.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    hearing_options = ["Normal", "Abnormal"]

    with c1:

        st.markdown("**Left ear hearing**")

        source_label(extracted.get("hearing_left"))

        left_default = (
            "Normal"
            if extracted.get("hearing_left") != "Abnormal"
            else "Abnormal"
        )

        hearing_left = st.selectbox(
            "Left Ear",
            hearing_options,
            index=hearing_options.index(left_default),
            key="review_hearing_left",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Right ear hearing**")

        source_label(extracted.get("hearing_right"))

        right_default = (
            "Normal"
            if extracted.get("hearing_right") != "Abnormal"
            else "Abnormal"
        )

        hearing_right = st.selectbox(
            "Right Ear",
            hearing_options,
            index=hearing_options.index(right_default),
            key="review_hearing_right",
            label_visibility="collapsed"
        )


    # ========================================================
    # BLOOD PRESSURE
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Blood Pressure</div>
        <div class="card-subtitle">
            Blood pressure values detected from the uploaded reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("**Systolic (mmHg)**")

        source_label(extracted.get("systolic"))

        systolic = st.number_input(
            "Systolic",
            min_value=50.0,
            max_value=250.0,
            value=float(extracted["systolic"])
            if extracted.get("systolic") is not None
            else 120.0,
            step=1.0,
            key="review_systolic",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Diastolic / Relaxation (mmHg)**")

        source_label(extracted.get("relaxation"))

        relaxation = st.number_input(
            "Diastolic",
            min_value=30.0,
            max_value=150.0,
            value=float(extracted["relaxation"])
            if extracted.get("relaxation") is not None
            else 80.0,
            step=1.0,
            key="review_relaxation",
            label_visibility="collapsed"
        )


    # ========================================================
    # BLOOD TEST
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Blood Test Results</div>
        <div class="card-subtitle">
            Values found in your uploaded laboratory reports.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Row 1

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**Fasting Blood Sugar (mg/dL)**")

        source_label(extracted.get("fasting_blood_sugar"))

        fasting_sugar = st.number_input(
            "Fasting Sugar",
            min_value=0.0,
            value=float(extracted["fasting_blood_sugar"])
            if extracted.get("fasting_blood_sugar") is not None
            else 90.0,
            step=0.5,
            key="review_fasting_sugar",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Total Cholesterol (mg/dL)**")

        source_label(extracted.get("cholesterol"))

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0.0,
            value=float(extracted["cholesterol"])
            if extracted.get("cholesterol") is not None
            else 190.0,
            step=1.0,
            key="review_cholesterol",
            label_visibility="collapsed"
        )

    with c3:

        st.markdown("**Triglyceride (mg/dL)**")

        source_label(extracted.get("triglyceride"))

        triglyceride = st.number_input(
            "Triglyceride",
            min_value=0.0,
            value=float(extracted["triglyceride"])
            if extracted.get("triglyceride") is not None
            else 120.0,
            step=1.0,
            key="review_triglyceride",
            label_visibility="collapsed"
        )


    # Row 2

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**HDL (mg/dL)**")

        source_label(extracted.get("HDL"))

        HDL = st.number_input(
            "HDL",
            min_value=0.0,
            value=float(extracted["HDL"])
            if extracted.get("HDL") is not None
            else 50.0,
            step=1.0,
            key="review_hdl",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**LDL (mg/dL)**")

        source_label(extracted.get("LDL"))

        LDL = st.number_input(
            "LDL",
            min_value=0.0,
            value=float(extracted["LDL"])
            if extracted.get("LDL") is not None
            else 100.0,
            step=1.0,
            key="review_ldl",
            label_visibility="collapsed"
        )

    with c3:

        st.markdown("**Hemoglobin (g/dL)**")

        source_label(extracted.get("hemoglobin"))

        hemoglobin = st.number_input(
            "Hemoglobin",
            min_value=0.0,
            value=float(extracted["hemoglobin"])
            if extracted.get("hemoglobin") is not None
            else 13.5,
            step=0.1,
            key="review_hemoglobin",
            label_visibility="collapsed"
        )


    # ========================================================
    # OTHER LAB INFORMATION
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Additional Laboratory Information</div>
        <div class="card-subtitle">
            Additional laboratory parameters used by the model.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("**Urine Protein**")

        source_label(extracted.get("urine_protein"))

        urine_protein = st.number_input(
            "Urine Protein",
            min_value=0.0,
            value=float(extracted["urine_protein"])
            if extracted.get("urine_protein") is not None
            else 0.0,
            step=0.1,
            key="review_urine_protein",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Serum Creatinine (mg/dL)**")

        source_label(extracted.get("serum_creatinine"))

        serum_creatinine = st.number_input(
            "Serum Creatinine",
            min_value=0.0,
            value=float(extracted["serum_creatinine"])
            if extracted.get("serum_creatinine") is not None
            else 0.8,
            step=0.1,
            key="review_creatinine",
            label_visibility="collapsed"
        )

    with c3:

        st.markdown("**AST (U/L)**")

        source_label(extracted.get("AST"))

        AST = st.number_input(
            "AST",
            min_value=0.0,
            value=float(extracted["AST"])
            if extracted.get("AST") is not None
            else 20.0,
            step=1.0,
            key="review_ast",
            label_visibility="collapsed"
        )

    c1, c2 = st.columns(2)

    with c1:

        st.markdown("**ALT (U/L)**")

        source_label(extracted.get("ALT"))

        ALT = st.number_input(
            "ALT",
            min_value=0.0,
            value=float(extracted["ALT"])
            if extracted.get("ALT") is not None
            else 20.0,
            step=1.0,
            key="review_alt",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**GTP (U/L)**")

        source_label(extracted.get("Gtp"))

        Gtp = st.number_input(
            "GTP",
            min_value=0.0,
            value=float(extracted["Gtp"])
            if extracted.get("Gtp") is not None
            else 30.0,
            step=1.0,
            key="review_gtp",
            label_visibility="collapsed"
        )


    # ========================================================
    # DENTAL
    # ========================================================

    st.markdown("""
    <div class="card">
        <div class="card-title">Dental Information</div>
        <div class="card-subtitle">
            These values may not be present in a standard blood report.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    yes_no = ["No", "Yes"]

    with c1:

        st.markdown("**Dental Caries**")

        source_label(extracted.get("dental_caries"))

        caries_default = (
            extracted["dental_caries"]
            if extracted.get("dental_caries") in yes_no
            else "No"
        )

        dental_caries = st.selectbox(
            "Dental Caries",
            yes_no,
            index=yes_no.index(caries_default),
            key="review_dental_caries",
            label_visibility="collapsed"
        )

    with c2:

        st.markdown("**Dental Tartar**")

        source_label(extracted.get("tartar"))

        tartar_default = (
            extracted["tartar"]
            if extracted.get("tartar") in yes_no
            else "No"
        )

        tartar = st.selectbox(
            "Dental Tartar",
            yes_no,
            index=yes_no.index(tartar_default),
            key="review_tartar",
            label_visibility="collapsed"
        )


    # ========================================================
    # CONFIRM
    # ========================================================

    st.markdown("""
    <div class="info-box">
        Please verify the green values before continuing.
        Yellow fields are values that were not detected in the uploaded reports.
    </div>
    """, unsafe_allow_html=True)


    if st.button(
        "Continue to Prediction",
        use_container_width=True,
        key="continue_prediction_button"
    ):

        # ----------------------------------------------------
        # Convert values into model format
        # ----------------------------------------------------

        gender_F = 1 if gender == "Female" else 0
        gender_M = 1 if gender == "Male" else 0

        tartar_N = 1 if tartar == "No" else 0
        tartar_Y = 1 if tartar == "Yes" else 0

        hearing_left_value = (
            1 if hearing_left == "Normal" else 0
        )

        hearing_right_value = (
            1 if hearing_right == "Normal" else 0
        )

        dental_caries_value = (
            1 if dental_caries == "Yes" else 0
        )

        # ----------------------------------------------------
        # Exact model dataframe
        # ----------------------------------------------------

        model_data = {

            "age": age,
            "height(cm)": height,
            "weight(kg)": weight,
            "waist(cm)": waist,

            "eyesight(left)": eyesight_left,
            "eyesight(right)": eyesight_right,

            "hearing(left)": hearing_left_value,
            "hearing(right)": hearing_right_value,

            "systolic": systolic,
            "relaxation": relaxation,

            "fasting blood sugar": fasting_sugar,
            "Cholesterol": cholesterol,
            "triglyceride": triglyceride,
            "HDL": HDL,
            "LDL": LDL,
            "hemoglobin": hemoglobin,

            "Urine protein": urine_protein,
            "serum creatinine": serum_creatinine,

            "AST": AST,
            "ALT": ALT,
            "Gtp": Gtp,

            "dental caries": dental_caries_value,

            "gender_F": gender_F,
            "gender_M": gender_M,

            "tartar_N": tartar_N,
            "tartar_Y": tartar_Y
        }

        # Keep exact training column order

        final_df = pd.DataFrame(
            [model_data],
            columns=feature_columns
        )

        st.session_state.final_data = final_df

        st.session_state.step = 3

        st.rerun()


# ============================================================
# STEP 3 — PREDICTION
# ============================================================

elif st.session_state.step == 3:

    show_steps(3)

    st.markdown("""
    <div class="card">
        <div class="card-title">Prediction</div>
        <div class="card-subtitle">
            The trained machine-learning model has analyzed the submitted
            health profile.
        </div>
    </div>
    """, unsafe_allow_html=True)

    final_df = st.session_state.final_data

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(final_df)

    result = int(prediction[0])

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if result == 1:

        st.markdown("""
        <div class="prediction-card">
            <div class="prediction-title">
                Predicted Smoking Status
            </div>
            <div class="prediction-value smoker">
                SMOKER
            </div>
            <div class="prediction-description">
                The trained machine-learning model classified the
                submitted health profile under the smoker category.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="prediction-card">
            <div class="prediction-title">
                Predicted Smoking Status
            </div>
            <div class="prediction-value non-smoker">
                NON-SMOKER
            </div>
            <div class="prediction-description">
                The trained machine-learning model classified the
                submitted health profile under the non-smoker category.
            </div>
        </div>
        """, unsafe_allow_html=True)


    # --------------------------------------------------------
    # Information used for prediction
    # --------------------------------------------------------

    with st.expander("View information used for prediction"):

        display_df = final_df.T.reset_index()

        display_df.columns = ["Parameter", "Value"]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # Start over
    # --------------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "Start New Analysis",
        use_container_width=True,
        key="new_analysis_button"
    ):

        for key in [
            "extracted",
            "report_text",
            "uploaded_names",
            "final_data"
        ]:

            if key in st.session_state:
                del st.session_state[key]

        st.session_state.step = 1

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    SmokeSense | Machine Learning Health Analysis Project
    <br>
    For educational and project purposes only
</div>
""", unsafe_allow_html=True)