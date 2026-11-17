
# ==========================================================
# STUDENT PLACEMENT ELIGIBILITY PREDICTION
# Streamlit Web Application
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "student_placement_dataset.csv"
MODEL_PATH = BASE_DIR / "placement_model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
IMPORTANCE_PATH = BASE_DIR / "feature_importance.csv"
RESULTS_PATH = BASE_DIR / "test_predictions.csv"


# ----------------------------------------------------------
# 2. LOAD PROJECT FILES
# ----------------------------------------------------------

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)

@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)

@st.cache_data
def load_test_results():
    return pd.read_csv(RESULTS_PATH)


# Check required files
required_files = [
    DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    IMPORTANCE_PATH,
    RESULTS_PATH
]

missing_files = [str(file.name) for file in required_files
                 if not file.exists()]

if missing_files:
    st.error("Some project files are missing:")
    for file in missing_files:
        st.write("- " + file)
    st.info(
        "Upload these files into the same GitHub folder "
        "as app.py."
    )
    st.stop()

try:
    df = load_dataset()
    model = load_model()
    metrics = load_metrics()
    importance_df = load_importance()
    test_results = load_test_results()

except Exception as error:
    st.error("Unable to load the project files.")
    st.exception(error)
    st.stop()


# ----------------------------------------------------------
# 3. SIDEBAR
# ----------------------------------------------------------

st.sidebar.title("🎓 Placement Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Overview",
        "Dataset Analysis",
        "Student Performance",
        "Model Evaluation",
        "Feature Importance",
        "Placement Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Machine Learning Project\n\n"
    "Algorithm: Random Forest Classifier\n\n"
    "Dataset: Synthetic student records"
)


# ----------------------------------------------------------
# 4. MAIN HEADER
# ----------------------------------------------------------

st.title("🎓 Student Placement Eligibility Prediction")
st.markdown(
    "### Machine Learning Based Placement Analytics Dashboard"
)

st.markdown(
    "Analyze student academic performance, skills and "
    "internship experience, and explore model-based placement predictions."
)

st.markdown("---")


# ----------------------------------------------------------
# 5. OVERVIEW PAGE
# ----------------------------------------------------------

if page == "Overview":

    st.subheader("📊 Project Overview")

    total_students = len(df)

    placed = int(
        (df["Placement_Status"] == "Placed").sum()
    )

    not_placed = int(
        (df["Placement_Status"] == "Not Placed").sum()
    )

    placement_rate = (
        placed / total_students * 100
        if total_students else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", f"{total_students:,}")
    col2.metric("Placed Students", f"{placed:,}")
    col3.metric("Not Placed", f"{not_placed:,}")
    col4.metric("Placement Rate", f"{placement_rate:.2f}%")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        status_counts = (
            df["Placement_Status"]
            .value_counts()
            .reset_index()
        )
        status_counts.columns = ["Placement Status", "Students"]

        fig = px.pie(
            status_counts,
            names="Placement Status",
            values="Students",
            hole=0.45,
            title="Placement Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        internship_counts = (
            df.groupby(["Internship", "Placement_Status"])
            .size()
            .reset_index(name="Students")
        )

        fig = px.bar(
            internship_counts,
            x="Internship",
            y="Students",
            color="Placement_Status",
            barmode="group",
            title="Internship Experience and Placement"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "Note: The dataset is synthetically generated "
        "for educational demonstration."
    )


# ----------------------------------------------------------
# 6. DATASET ANALYSIS PAGE
# ----------------------------------------------------------

elif page == "Dataset Analysis":

    st.subheader("📁 Dataset Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Records", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.markdown("### Dataset Information")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values
        }),
        use_container_width=True
    )

    st.markdown("### Statistical Summary")
    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )

    st.markdown("### Filter Student Records")

    status_filter = st.selectbox(
        "Select Placement Status",
        ["All", "Placed", "Not Placed"]
    )

    if status_filter == "All":
        filtered_df = df
    else:
        filtered_df = df[
            df["Placement_Status"] == status_filter
        ]

    st.write("Filtered records:", len(filtered_df))
    st.dataframe(filtered_df, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Dataset",
        data=csv_data,
        file_name="filtered_student_dataset.csv",
        mime="text/csv"
    )


# ----------------------------------------------------------
# 7. STUDENT PERFORMANCE PAGE
# ----------------------------------------------------------

elif page == "Student Performance":

    st.subheader("📈 Student Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            df,
            x="CGPA",
            color="Placement_Status",
            nbins=20,
            title="CGPA Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            df,
            x="Placement_Status",
            y="Technical_Skills_Score",
            color="Placement_Status",
            title="Technical Skills Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig = px.box(
            df,
            x="Placement_Status",
            y="Communication_Skills_Score",
            color="Placement_Status",
            title="Communication Skills Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df,
            x="Backlogs",
            color="Placement_Status",
            barmode="group",
            title="Backlogs and Placement"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Academic Performance Comparison")

    academic_columns = [
        "10th_Percentage",
        "Intermediate_Diploma_Percentage"
    ]

    academic_summary = (
        df.groupby("Placement_Status")[academic_columns]
        .mean()
        .reset_index()
        .melt(
            id_vars="Placement_Status",
            var_name="Academic Qualification",
            value_name="Average Percentage"
        )
    )

    fig = px.bar(
        academic_summary,
        x="Academic Qualification",
        y="Average Percentage",
        color="Placement_Status",
        barmode="group",
        title="Average Academic Scores by Placement Status"
    )

    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------
# 8. MODEL EVALUATION PAGE
# ----------------------------------------------------------

elif page == "Model Evaluation":

    st.subheader("🤖 Random Forest Model Evaluation")

    st.write(
        "The model was evaluated using a separate test dataset."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%"
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1_score'] * 100:.2f}%"
    )

    st.markdown("---")

    st.write("Training records:", metrics["training_records"])
    st.write("Testing records:", metrics["testing_records"])

    if (
        "Actual_Status" in test_results.columns
        and "Predicted_Status" in test_results.columns
    ):

        labels = ["Not Placed", "Placed"]

        cm = confusion_matrix(
            test_results["Actual_Status"],
            test_results["Predicted_Status"],
            labels=labels
        )

        fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Status",
            yaxis_title="Actual Status"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Test Predictions")
        st.dataframe(
            test_results.head(20),
            use_container_width=True
        )

    st.warning(
        "These results measure performance on synthetic data. "
        "They do not establish real-world placement accuracy."
    )


# ----------------------------------------------------------
# 9. FEATURE IMPORTANCE PAGE
# ----------------------------------------------------------

elif page == "Feature Importance":

    st.subheader("⭐ Feature Importance Analysis")

    importance_sorted = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance_sorted,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        text_auto=".3f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature Importance Table")
    st.dataframe(
        importance_df.sort_values(
            "Importance", ascending=False
        ),
        use_container_width=True
    )

    st.info(
        "Feature importance indicates how much a feature "
        "contributes to the fitted Random Forest model. "
        "It does not prove that the feature causes placement."
    )


# ----------------------------------------------------------
# 10. INTERACTIVE PLACEMENT PREDICTION
# ----------------------------------------------------------

elif page == "Placement Prediction":

    st.subheader("🔍 Student Placement Prediction")

    st.write(
        "Enter the student's details below to obtain a "
        "prediction from the trained Random Forest model."
    )

    with st.form("placement_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            cgpa = st.slider(
                "CGPA",
                min_value=4.5,
                max_value=10.0,
                value=8.0,
                step=0.1
            )

            tenth = st.slider(
                "10th Percentage",
                min_value=45,
                max_value=100,
                value=80
            )

            intermediate = st.slider(
                "Intermediate/Diploma Percentage",
                min_value=45,
                max_value=100,
                value=80
            )

            technical = st.slider(
                "Technical Skills Score",
                min_value=30,
                max_value=100,
                value=75
            )

        with col2:
            communication = st.slider(
                "Communication Skills Score",
                min_value=30,
                max_value=100,
                value=75
            )

            backlogs = st.selectbox(
                "Number of Backlogs",
                [0, 1, 2, 3, 4]
            )

            internship = st.selectbox(
                "Internship Experience",
                ["Yes", "No"]
            )

        submitted = st.form_submit_button(
            "Predict Placement",
            use_container_width=True
        )

    if submitted:

        student_data = pd.DataFrame([{
            "CGPA": cgpa,
            "10th_Percentage": tenth,
            "Intermediate_Diploma_Percentage": intermediate,
            "Technical_Skills_Score": technical,
            "Communication_Skills_Score": communication,
            "Backlogs": backlogs,
            "Internship": internship
        }])

        try:
            prediction = model.predict(student_data)[0]

            probabilities = model.predict_proba(student_data)[0]

            classes = list(
                model.named_steps["classifier"].classes_
            )

            placed_probability = probabilities[
                classes.index("Placed")
            ]

            st.markdown("---")
            st.subheader("Prediction Result")

            if prediction == "Placed":
                st.success("Predicted Status: PLACED")
            else:
                st.error("Predicted Status: NOT PLACED")

            st.metric(
                "Estimated Placed Probability",
                f"{placed_probability * 100:.2f}%"
            )

            probability_df = pd.DataFrame({
                "Status": classes,
                "Probability": probabilities
            })

            fig = px.bar(
                probability_df,
                x="Status",
                y="Probability",
                color="Status",
                range_y=[0, 1],
                title="Prediction Probability by Class"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.caption(
                "This is a model prediction, not a guarantee "
                "of employment or an actual recruitment decision. "
                "The model uses synthetic training data."
            )

        except Exception as error:
            st.error("Prediction could not be completed.")
            st.exception(error)


# ----------------------------------------------------------
# 11. FOOTER
# ----------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:gray;">
        Student Placement Eligibility Prediction Using Machine Learning
        <br>
        Developed using Python, Streamlit and Random Forest
        <br>
        Academic Demonstration Project
    </div>
    """,
    unsafe_allow_html=True
)
