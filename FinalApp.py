# Importing Libraries
import pandas as pd
import warnings
from gensim.parsing.preprocessing import remove_stopwords, stem_text  # Import remove_stopwords and stem_text functions
import re  # regular expression
import streamlit as st
import pickle
import docx2txt
from PyPDF2 import PdfFileReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np  # Import numpy for array manipulation

# Suppress warnings
warnings.filterwarnings("ignore")

classification_model = pickle.load(open("finalmodel_xgboost.pkl", 'rb'))
print('Loaded model from file')

# Load category-wise words from category_word_frequencies.pkl
with open("category_word_frequencies.pkl", "rb") as f:
    category_word_frequencies = pickle.load(f)

# Function to preprocess text
def preprocess(x):
    # Removing both the leading and the trailing characters such as spaces
    x = x.strip()
    
    # Removing spaces in between
    x = re.sub(r'\s+', " ", x)
    
    # To remove special characters and numbers, allow a-z A-Z and space character only
    x = re.sub('[^a-zA-Z ]', '', x)
    
    # Converting into lowercase characters
    x = x.lower()
    
    # Splitting
    x = x.split()
    
    # Remove stopwords using gensim's remove_stopwords function
    x = " ".join([remove_stopwords(word) for word in x])
    
    # Stemming
    x = stem_text(x)
    
    return x

# Function to classify a resume based on category-wise words
def classify_resume(text):
    text = preprocess(text)
    category_scores = {category: 0 for category in category_word_frequencies.keys()}
    
    for category, keywords in category_word_frequencies.items():
        for keyword in keywords:
            if keyword in text:
                category_scores[category] += 1
                
    predicted_category = max(category_scores, key=category_scores.get)
    return predicted_category

def main():
    # Giving a title
    st.title('Resume Classification Application')
    resume_text = []
    file_name = []
    uploaded_files = st.file_uploader("Choose a .docx file", type=['docx', 'pdf'], accept_multiple_files=True)
    
    submit = st.button('Predict Resume class/category')
    if submit:
        if len(uploaded_files) > 0:
            for uploaded_file in uploaded_files:
                # File Name
                if uploaded_file.type == "application/pdf":
                    pdfReader = PdfFileReader(uploaded_file)
                    count = pdfReader.numPages
                    text = ""
                    for i in range(count):
                        page = pdfReader.getPage(i)
                        text += page.extractText()
                else:
                    # Reading the document text
                    text = docx2txt.process(uploaded_file)
                file_name.append(uploaded_file.name)
                resume_text.append(text)
            
            # Creating DataFrame for uploaded resume
            resume_data = pd.DataFrame()
            resume_data['File_Name'] = file_name
            resume_data['Resume_Text'] = resume_text
            
            # Predict categories for each resume
            resume_data['Category'] = resume_data['Resume_Text'].apply(classify_resume)
            
            st.subheader("Resume Category:")
            for index, row in resume_data.iterrows():  
                st.write(row['File_Name'], ':', row['Category'])
            st.write('Note: This application classifies Resumes into Categories based on extracted words.')

        elif len(uploaded_files) == 0:
            st.write('Upload a file first!')
        else:
            st.session_state["upload_state"] = "Upload a file first!"

if __name__ == '__main__':
    main()
