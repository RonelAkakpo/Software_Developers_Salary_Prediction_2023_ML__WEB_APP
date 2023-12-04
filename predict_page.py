import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
       data = pickle.load(file)
    return data

data = load_model()

rf_loaded1 = data["model"]    
countr = data["countr"]
educ = data["educ"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")
    
    
    
    countries = (
          "United States of America",                       
          "Germany",                                                  
          "United Kingdom of Great Britain and Northern Ireland",    
          "Canada",                                                   
          "India",                                                    
          "France",                                                   
          "Netherlands",                                              
          "Australia",                                                
          "Brazil",                                                    
          "Spain",                                                     
          "Sweden",                                                    
          "Italy",                                                     
          "Poland",                                                 
          "Switzerland",                                              
          "Denmark",                                                   
          "Norway",                                                
          "Israel",
         )
    
    
    education = (
        "less than a Bachelors", 
        "Bachelor’s degree", 
        "Master’s degree",
        "Post Grad",
        )
    
    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    
    experience = st.slider("Years of Experience", 0, 50, 3)
    
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience ]])
        X[:,0] = countr.transform(X[:,0])
        X[:,1] = educ.transform(X[:,1])
        X = X.astype(float)
        
        salary = rf_loaded1.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")