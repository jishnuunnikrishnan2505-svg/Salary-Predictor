import streamlit as st
import pandas as pd
from sklearn import linear_model
from word2number import w2n

# Title of the app
st.title("Salary Prediction App")

# 1. Load and Preprocess Data (Logic from your notebook)
@st.cache_data
def load_data():
    df = pd.read_csv('hiring.csv')
    # Fill missing experience with 'zero'
    df.experience = df.experience.fillna('zero')
    # Convert word to number
    df.experience = df.experience.apply(w2n.word_to_num)
    # Fill missing test scores with mean
    df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean())
    # Rename for easier access
    df = df.rename(columns={'salary($)': 'salary'})
    return df

df = load_data()

# 2. Train the Model
reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df.salary)

# 3. Streamlit User Interface for Inputs
st.subheader("Enter Candidate Details")

experience = st.number_input("Experience (Years)", min_value=0, max_value=50, value=2)
test_score = st.slider("Test Score (out of 10)", 0.0, 10.0, 8.0)
interview_score = st.slider("Interview Score (out of 10)", 0.0, 10.0, 7.0)

# 4. Prediction Button
if st.button("Predict Salary"):
    prediction = reg.predict([[experience, test_score, interview_score]])
    st.success(f"The estimated salary for this candidate is: ${prediction[0]:,.2f}")

# Optional: Show the dataset
if st.checkbox("Show Training Data"):
    st.write(df)