
import streamlit as st
import pandas as pd
import joblib
from keras.models import load_model
from PIL import Image
import numpy as np
import librosa
import tempfile
import os
import cv2

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="MediSense AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #173b57, #286b8f);
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        font-size: 1.1rem;
        margin-bottom: 0;
    }

    .feature-card {
        padding: 1.2rem;
        border-radius: 15px;
        background-color: white;
        border: 1px solid #e2e8f0;
        min-height: 150px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .feature-card h3 {
        color: #173b57;
        font-size: 1.1rem;
    }

    .feature-card p {
        color: #52606d;
        font-size: 0.92rem;
    }

    .section-heading {
        color: #173b57;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .disclaimer {
        padding: 1rem;
        border-radius: 12px;
        background-color: #fff7e6;
        border: 1px solid #f2c879;
        color: #765313;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load trained models
# --------------------------------------------------

@st.cache_resource
def load_all_models():
    heart_model = joblib.load("models/heart_disease_model.pkl")
    xray_model = load_model("models/xray_pneumonia_model.keras")
    audio_model = joblib.load("models/audio_heartbeat_model.pkl")
    video_model = load_model("models/video_activity_model.keras")

    return heart_model, xray_model, audio_model, video_model


model, xray_model, audio_model, video_model = load_all_models()


# --------------------------------------------------
# Helper function for video processing
# --------------------------------------------------

def extract_video_frames(video_path, frame_count=10):
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        frame_count,
        dtype=int
    )

    frames = []

    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128))
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    return np.array(frames)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/medical-doctor.png",
        width=75
    )

    st.title("MediSense AI")
    st.caption("Smart Patient Monitoring System")

    st.markdown("---")

    selected_module = st.radio(
        "Choose an AI module",
        [
            "Dashboard Overview",
            "Heart Disease Risk",
            "Chest X-ray Analysis",
            "Heartbeat Audio",
            "Human Activity"
        ]
    )

    st.markdown("---")
    st.caption("Built with Python, TensorFlow, Scikit-learn and Streamlit")


# --------------------------------------------------
# Main header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🩺 MediSense AI</h1>
        <p>An integrated AI portfolio for healthcare-related data analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Dashboard Overview
# --------------------------------------------------

if selected_module == "Dashboard Overview":

    st.markdown(
        '<h2 class="section-heading">Project Overview</h2>',
        unsafe_allow_html=True
    )

    st.write(
        "MediSense AI combines four machine-learning modules into one "
        "interactive dashboard. Select a module from the sidebar to test "
        "its trained model."
    )

    st.markdown("### Available AI Modules")

    card_col1, card_col2, card_col3, card_col4 = st.columns(4)

    with card_col1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>❤️ Heart Risk</h3>
                <p>Predicts a model-based heart disease risk classification using patient health information.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_col2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🫁 Chest X-ray</h3>
                <p>Classifies a chest X-ray image as NORMAL or PNEUMONIA.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_col3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🔊 Heartbeat Audio</h3>
                <p>Analyzes heartbeat audio features and identifies normal or abnormal patterns.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with card_col4:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🎥 Human Activity</h3>
                <p>Recognizes Walking With Dog and Bodyweight Squats from video.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### System Workflow")

    workflow_col1, workflow_col2, workflow_col3 = st.columns(3)

    with workflow_col1:
        st.info("**1. Upload or enter data**\n\nProvide patient information, an image, audio, or video.")

    with workflow_col2:
        st.info("**2. Run the trained model**\n\nThe selected machine-learning model processes the input.")

    with workflow_col3:
        st.info("**3. View the result**\n\nThe dashboard displays the model prediction and score.")


# --------------------------------------------------
# Heart Disease Risk Module
# --------------------------------------------------

elif selected_module == "Heart Disease Risk":

    st.markdown(
        '<h2 class="section-heading">❤️ Heart Disease Risk Prediction</h2>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter patient information to generate a model-based heart disease risk classification."
    )

    with st.form("heart_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=50
            )

            sex = st.selectbox(
                "Sex",
                ["Male", "Female"]
            )

            cp = st.selectbox(
                "Chest Pain Type",
                [
                    "typical angina",
                    "atypical angina",
                    "non-anginal",
                    "asymptomatic"
                ]
            )

            trestbps = st.number_input(
                "Resting Blood Pressure",
                min_value=50.0,
                max_value=250.0,
                value=120.0
            )

            chol = st.number_input(
                "Cholesterol",
                min_value=50.0,
                max_value=700.0,
                value=200.0
            )

        with col2:
            fbs = st.checkbox("Fasting Blood Sugar > 120 mg/dl")

            restecg = st.selectbox(
                "Resting ECG",
                [
                    "normal",
                    "ST-T wave abnormality",
                    "left ventricular hypertrophy"
                ]
            )

            thalch = st.number_input(
                "Maximum Heart Rate",
                min_value=50.0,
                max_value=250.0,
                value=150.0
            )

            exang = st.checkbox("Exercise-Induced Angina")

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=10.0,
                value=1.0
            )

            slope = st.selectbox(
                "ST Segment Slope",
                [
                    "upsloping",
                    "flat",
                    "downsloping"
                ]
            )

        submitted = st.form_submit_button(
            "🔍 Predict Heart Disease Risk",
            use_container_width=True
        )

    if submitted:

        patient_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalch": thalch,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope
        }])

        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1]

        st.markdown("### Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if prediction == 1:
                st.error("⚠️ Higher heart disease risk classification")
            else:
                st.success("✅ Lower heart disease risk classification")

        with result_col2:
            st.metric(
                "Estimated model probability",
                f"{probability:.1%}"
            )


# --------------------------------------------------
# Chest X-ray Module
# --------------------------------------------------

elif selected_module == "Chest X-ray Analysis":

    st.markdown(
        '<h2 class="section-heading">🫁 Chest X-ray Pneumonia Detection</h2>',
        unsafe_allow_html=True
    )

    st.write("Upload a chest X-ray image for classification.")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        xray_image = Image.open(uploaded_file).convert("L")

        image_col1, image_col2 = st.columns(2)

        with image_col1:
            st.image(
                xray_image,
                caption="Uploaded Chest X-ray",
                use_container_width=True
            )

        with image_col2:
            resized_image = xray_image.resize((224, 224))
            image_array = np.array(resized_image) / 255.0
            image_array = np.expand_dims(image_array, axis=-1)
            image_array = np.expand_dims(image_array, axis=0)

            probability = xray_model.predict(
                image_array,
                verbose=0
            )[0][0]

            st.markdown("### Prediction Result")

            if probability >= 0.5:
                st.error("⚠️ Model Prediction: PNEUMONIA")
            else:
                st.success("✅ Model Prediction: NORMAL")

            st.metric(
                "Model probability for pneumonia",
                f"{probability:.1%}"
            )


# --------------------------------------------------
# Heartbeat Audio Module
# --------------------------------------------------

elif selected_module == "Heartbeat Audio":

    st.markdown(
        '<h2 class="section-heading">🔊 Heartbeat Audio Screening</h2>',
        unsafe_allow_html=True
    )

    st.write("Upload a heartbeat audio file in MP3 or WAV format.")

    uploaded_audio = st.file_uploader(
        "Upload heartbeat audio",
        type=["mp3", "wav"]
    )

    if uploaded_audio is not None:

        st.audio(uploaded_audio)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_audio.name)[1]
        ) as temp_audio:

            temp_audio.write(uploaded_audio.getbuffer())
            temp_audio_path = temp_audio.name

        try:
            audio, sample_rate = librosa.load(
                temp_audio_path,
                sr=22050,
                mono=True
            )

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=13
            )

            audio_features_input = np.concatenate([
                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1)
            ])

            feature_columns = (
                [f"mfcc_mean_{i}" for i in range(1, 14)]
                + [f"mfcc_std_{i}" for i in range(1, 14)]
            )

            audio_input_df = pd.DataFrame(
                [audio_features_input],
                columns=feature_columns
            )

            audio_prediction = audio_model.predict(
                audio_input_df
            )[0]

            audio_probability = audio_model.predict_proba(
                audio_input_df
            )[0][1]

            st.markdown("### Prediction Result")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if audio_prediction == 1:
                    st.warning("⚠️ Abnormal heartbeat pattern")
                else:
                    st.success("✅ Normal heartbeat pattern")

            with result_col2:
                st.metric(
                    "Abnormal pattern score",
                    f"{audio_probability:.1%}"
                )

        except Exception as error:
            st.error(f"Audio processing failed: {error}")

        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)


# --------------------------------------------------
# Human Activity Module
# --------------------------------------------------

elif selected_module == "Human Activity":

    st.markdown(
        '<h2 class="section-heading">🎥 Human Activity Recognition</h2>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a video to classify it as Walking With Dog or Bodyweight Squats."
    )

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["avi", "mp4", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_video.name)[1]
        ) as temp_video:

            temp_video.write(uploaded_video.read())
            temp_video_path = temp_video.name

        try:
            frames = extract_video_frames(
                temp_video_path,
                frame_count=10
            )

            if frames is None or len(frames) != 10:
                st.error("Could not extract 10 frames from this video.")

            else:
                frames = frames.astype("float32") / 255.0
                frames = np.expand_dims(frames, axis=0)

                prediction = video_model.predict(
                    frames,
                    verbose=0
                )[0][0]

                if prediction >= 0.5:
                    activity = "Walking With Dog"
                else:
                    activity = "Bodyweight Squats"

                result_col1, result_col2 = st.columns(2)

                with result_col1:
                    st.success(f"Predicted activity: {activity}")

                with result_col2:
                    st.metric(
                        "Model score",
                        f"{prediction:.2%}"
                    )

        except Exception as error:
            st.error(f"Video processing failed: {error}")

        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important disclaimer:</strong>
        This is an educational and portfolio project. The predictions are
        generated by experimental machine-learning models and must not be
        treated as medical diagnoses, clinical decisions, or professional
        medical advice.
    </div>
    """,
    unsafe_allow_html=True
)