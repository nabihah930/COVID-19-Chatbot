import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle

lemmatizer = WordNetLemmatizer()
#intents_file = open('C:/Users/ACER/OneDrive/Desktop/AI-Project/intent.json').read()            #loading our file that has all the intents
intents_file = open('intent.json', encoding="utf8").read()
intents = json.loads(intents_file)                                                              #have to add encoding wll not work without it.
#intents = json.loads('intent.json')
print(f"Intents:\n{intents}")

words=[]                                                                                        #this is the vocabulary for the bot
classes = []                                                                                    #this is the list of entities
documents = []
ignore_symbols = ['!', '?', ',', '.']                                                           #these act as stop words we use in IR
for intent in intents['intents']:
    for pattern in intent['patterns']:
        token = nltk.word_tokenize(pattern)                                                     #tokenizing
        words.extend(token)
        documents.append((token, intent['tag']))                                                #add documents in the corpus
        if intent['tag'] not in classes:                                                        #add to our classes list
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_symbols]             #lemmaztization
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
print(f"Documents({len(documents)}):\n{documents}")
print(f"Classes/entity({len(classes)}):\n{classes}")
print(f"Vocabulary({len(words)}):\n{words}")
pickle.dump(words,open('words.pkl','wb'))                                                       #saving our vocabulary and our classes
pickle.dump(classes,open('classes.pkl','wb'))


#CREATION OF TRAINING DATA
print("Creating training data...")
training = []
output_empty = [0]*len(classes)                                                               #array if length equal to size of classes for the output
for doc in documents:
    bag = []                                                                                    #bag-of-words concept same as in IR
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]              #same preprocessing steps that we did earlier
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)                               #if word present->1
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])                                                                   #x->patterns & y->intents
train_y = list(training[:,1])
print("Training data: CREATED")

#NOW THAT WE HAVE CREATED THE TRAINING DATA SET WE WILL CREATE THE MODEL
print("Creating model...")
model = Sequential()                                                                            #using sequential model for a simple stack of layers
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))                        #adding layers to the model incrementally
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])             #Now to train our model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Model: CREATED")