import tkinter
from tkinter import *
import pickle
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json', encoding="utf8").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)                                                       #apply same preprocessing on sentence as we did to
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]                    #create the vocabulary
    return sentence_words

def bag_of_words(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                bag[i] = 1                                                                              #1->word present
                if show_details:
                    print ("Present in bag of words: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

#Creating graphical interface
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':                                                                                       #if no message
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
main_window = Tk()
main_window.title("COVID-19 Chatbot")
main_window.geometry("500x500")
main_window.resizable(width=FALSE, height=FALSE)
#creating layout now
ChatBox = Text(main_window, bd=0, bg="white", height="10", width="50", font="Arial",)
ChatBox.config(state=DISABLED)
scrollbar = Scrollbar(main_window, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set
#creating button and box to enter and send message
button = Button(main_window, font=("Verdana",12,'bold'), text="Send", width="12", height=5,bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',command= send )
EntryBox = Text(main_window, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
scrollbar.place(x=376,y=6, height=386)                                                              #placing all the elements
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
button.place(x=6, y=401, height=90)
main_window.mainloop()