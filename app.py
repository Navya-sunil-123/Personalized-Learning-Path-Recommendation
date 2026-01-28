import streamlit as st
import numpy as np
import pickle

# Load trained objects
model=pickle.load(open("model.pkl", "rb"))
scaler=pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("Personalized Learning Path Recommendation")

# Inputs
age=st.number_input("Age", 18, 60, 25)
experience=st.number_input("Experience (Years)", 0, 20, 2)
avg_score=st.slider("Average Score (%)", 0, 100, 60)
time_spent=st.number_input("Time Spent Learning (hrs/week)", 1.0, 50.0, 10.0)
completed=st.number_input("Completed Courses", 0, 50, 2)

education=st.selectbox("Education Level", ["UG", "PG", "PhD"])
domain=st.selectbox("Interest Domain", ["AI", "Web Development", "Data Science"])

# mappings
edu_map={"UG": 0, "PG": 1, "PhD": 2}
domain_map={"AI": 0, "Web Development": 1, "Data Science": 2}

# Prediction 
if st.button("Predict Learning Path"):
    
    sample=np.array([[
        age,
        experience,
        avg_score,
        time_spent,
        completed,
        edu_map[education],
        domain_map[domain]
    ]])

    # Scale , predict
    sample_scaled=scaler.transform(sample)
    prediction=model.predict(sample_scaled)[0]

    # Show prediction
    st.success(f"Predicted Level: {prediction}")

    # Recommended courses
    if prediction=="Beginner":
        courses=["Python Basics", "Introduction to Programming", "Math for Machine Learning"]
    elif prediction=="Intermediate":
        courses=["Machine Learning Algorithms", "Data Analysis with Pandas", "SQL for Data Science"]
    else:
        courses=["Deep Learning", "MLOps Fundamentals", "Advanced NLP / Computer Vision"]

    st.subheader("Recommended Courses")
    for c in courses:
        st.write("•", c)

    


