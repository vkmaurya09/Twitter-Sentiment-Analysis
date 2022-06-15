import streamlit as st
import pickle
from preprocessing import preprocess
from nltk.stem import WordNetLemmatizer

def load_models():
    '''
    Replace '..path/' by the path of the saved models.
    '''
    
    # Load the vectoriser.
    file = open('vectoriser_1_2.pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open('Sentiment-Svm.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    clean_text = preprocess(text)
    if clean_text=='':
        return "There is no proper text" 
    textdata = vectoriser.transform(clean_text)
    sentiment = model.predict(textdata)
    # return sentiment
    # print(sentiment)
    result  = "Positive" if sentiment==1 else "Negative"
    
    return result

if __name__=="__main__":
    # Loading the models.
    vectoriser, LRmodel = load_models()
    
    # Text to classify should be in a list.
    # text = st.text_input('Enter Text')
    

    form = st.form(key='my-form')
    text = form.text_input("Enter Text")
    submit = form.form_submit_button("Submit")
    if submit:
        result = predict(vectoriser, LRmodel, [text])
        st.write(f"Model Predicted : {result}")