import datetime
import os
import threading
import nltk
import random
import csv
import streamlit as st
import ssl
import json
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

#load the intents file 
file_path = os.path.abspath("./intent.json")
with open(file_path , "r") as file:
    intents = json.load(file)
print(file_path)


# Initiaze the vectorizer and classifier
vectorizer  = TfidfVectorizer()
clf = LogisticRegression(random_state=0 , max_iter=10000)

#Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

#training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x , y)


# Chatbot

def chatBot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            respose = random.choice(intent['responses'])
            return respose

def speak(response):
    engine = pyttsx3.init()
    engine.say("Hello! I am your chatbot.")
    engine.setProperty("rate", 150)  # Speed of speech
    engine.setProperty("volume", 1)  # Volume (0.0 to 1.0)
    voices = engine.getProperty('voices')
    engine.setProperty('voice' , voices[1].id)
    engine.say(response)
    engine.runAndWait()        
        
counter = 0

def main():
    global counter
    st.title("Welcome to the multi response chatbot")


    st.markdown(
    """
    <style>
    /* Style for text input */
    .stTextInput>div>div>input {
       
        
        background-color: #f0f8ff; /* Light blue */
        border: 2px solid #4CAF50; /* Green border */
        border-radius: 10px;
        padding: 10px;
        font-size: 20px;
        max-height: 200px;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    

    # create a sideBar menu  with optionss 
    menu = ["Home" , "Conversation History" , "About"]
    choice = st.sidebar.selectbox("Menu" , menu)

   

    #Home Menu
    if choice == "Home":
        st.write("Having to great here, Plese type a message and press Enter to start")

        #check if the  csf file exists  or not
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv' , 'w' , newline='' , encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)      
                csv_writer.writerow(['User Input' , 'Chatbot Response' , 'Timestamp'])

        counter += 1

        user_input = st.text_input("You: " , key=f"user_input_{counter}")



        if user_input:

            # Convert the user Input to a string
            user_input_str = str(user_input)

            response = chatBot(user_input)
            thread = threading.Thread(target=speak, args=(response,))
            thread.start()

            #st.text_area("Chatbot: " , value=response , height=120, max_chars=None , key=f"chatbot_response_{counter}")

            #get the current time Stamp
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")

            with open('chat_log.csv' , 'a' , newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input_str, response , timestamp])

            if response.lower() in ['goodbye' , 'bye']:
                st.write("Thank you for chatting with me. have a great day")
                st.stop()

    elif choice ==  "Conversation History":
        #Display the conversation history
        st.header("Conversation History")
        
        with open('chat_log.csv' , 'r' ,encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)

            for row in csv_reader:
                st.text(f"User:  {row[0]}")
                st.text(f"Chatbot:  {row[1]}")
                st.text(f"Timestamp:  {row[2]}")
                st.markdown("----")

    elif choice == "About":
        st.write("The goal of this chatbot is to respond and understand the user input on intents")
        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression.
        2. For building the chatbot interface using streamlit as a web framework.                  
        """)

        st.subheader("Dateset: ")
        st.write("""
        - Intents : The intent of the user Input
        - Entities : The entities extracted from user
        - Text : The user input text                  
        """)

        st.subheader("Streamlit Chaatbot InterFace: ")
        st.write("The chatbot interface is built using streamlit")

        st.subheader("Conclusion:")
        st.write("In this project, a chatbot is built that can understand and respond to user Input")


if __name__ == '__main__':
    main()        

                           
