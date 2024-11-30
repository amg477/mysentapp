import pandas as pd
import numpy as np
from plotnine import * 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import streamlit as st

ss = pd.read_csv('social_media_usage.csv')

def clean_sm(x): 
    return np.where(x == 1, 1, 0)

ss = (ss.loc[
        (ss['income'].between(1, 9)) & (ss['educ2'].between(1, 8)) & 
        (ss['par'] <= 2) & (ss['marital'] <= 2) & (ss['gender'] <= 2) & (ss['age'] <= 98)]
    .assign(
        sm_li=ss['web1h'].apply(clean_sm),
        parent=ss['par'].apply(clean_sm),
        married=ss['marital'].apply(clean_sm),
        female=ss['gender'].apply(lambda x: 1 if x == 2 else 0))
    .filter(items=['income', 'educ2', 'parent', 'married', 'female', 'age', 'sm_li'])
    .dropna())

ss['sm_li'] = ss['sm_li'].astype('category')
ss['educ2'] = ss['educ2'].astype('category')
ss['parent'] = ss['parent'].astype('category')
ss['married'] = ss['married'].astype('category')
ss['female'] = ss['female'].astype('category')

y = ss['sm_li']
x = ss.drop("sm_li", axis =1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=16)

logreg = LogisticRegression(class_weight = "balanced", random_state = 16)
logreg.fit(x_train, y_train) 

st.markdown("Welcome to my app!")

st.markdown("Please provide your personal details below to predict your usage of LinkedIn:")

st.subheader("Do you ever use LinkedIn?")
sm_li = st.selectbox(
    "Select an option:",
    options=[
        (1, "Yes, I use LinkedIn"),
        (0, "No, I do not use LinkedIn")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your household income?")
income = st.selectbox(
    "Select your household income range:",
    options=[
        (1, "Less than $10,000"),
        (2, "$10,000 - $19,999"),
        (3, "$20,000 - $29,999"),
        (4, "$30,000 - $39,999"),
        (5, "$40,000 - $49,999"),
        (6, "$50,000 - $74,999"),
        (7, "$75,000 - $99,999"),
        (8, "$100,000 - $149,999"),
        (9, "$150,000 or more"),
        (10, "Don't Know"),
        (11, "Refused")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your highest level of education?")
educ2 = st.selectbox(
    "Select your highest level of education:",
    options=[
        (1, "Less than high school"),
        (2, "High school incomplete"),
        (3, "High school graduate"),
        (4, "Some college, no degree"),
        (5, "Two-year associate degree"),
        (6, "Four-year bachelor's degree"),
        (7, "Some postgraduate schooling, no degree"),
        (8, "Postgraduate or professional degree")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("Are you a parent of a child under 18 living in your home?")
parent = st.radio(
    "Select your parental status:",
    options=[
        (1, "Yes"),
        (0, "No")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your marital status?")
married = st.radio(
    "Select your marital status:",
    options=[
        (1, "Married"),
        (0, "Not Married")
    ],
    format_func=lambda x: x[1]
)[0]

st.subheader("What is your gender?")
female = st.radio(
    "Select your gender:",
    options=[
        (1, "Female"),
        (0, "Male")
    ],
    format_func=lambda x: x[1]
)[0]

age = st.number_input("Enter your age:", "")

def sent_app(user_data):

    user_data = pd.DataFrame([user_data], columns=['income', 'educ2', 'parent','married', 'female', 'age'])

    probability = logreg.predict_proba(user_data)[0][1]
    classification = logreg.predict(user_data)[0]
    
    st.subheader("Results:")
    st.write(f"**Classification:** You are {'a LinkedIn user' if classification == 1 else 'not a LinkedIn user'}.")
    st.write(f"**Probability:** There is a {probability * 100:.2f}% chance that you are a LinkedIn user.")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100, 
        title={'text': f"LinkedIn User Probability: {probability * 100:.2f}%"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 30], "color": "mediumvioletred"},
                {"range": [30, 70], "color": "darkturquoise"},
                {"range": [70, 100], "color": "mediumspringgreen"},],
            "bar": {"color": "gold"} }))
    return st.plotly_chart(fig)

if st.button("Submit"):
    user_data = {
        'income': income,
        'educ2': educ2,
        'parent': parent,
        'married': married,
        'female': female,
        'age': age
    }
    sent_app(user_data)
