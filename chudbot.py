import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import json
import pickle
import random

# Download NLTK data
nltk.download("punkt")

# Load intents from a JSON file
with open('brain.json') as intents:
    data = json.load(intents)

stemmer = LancasterStemmer()

try:
    # Try to load preprocessed data if available
    with open('data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    # If preprocessed data is not available, preprocess the data
    words = []
    labels = []
    x_docs = []
    y_docs = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            x_docs.append(wrds)
            y_docs.append(intent['tag'])

            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    # One-hot encoding, converting the words to numerical values
    for x, doc in enumerate(x_docs):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(y_docs[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# Define the neural network architecture
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    # Try to load the trained model if available
    model.load("model.tflearn")
except:
    # If the model is not available, train it
    model.fit(training, output, n_epoch=100, batch_size=8, show_metric=True)
    model.save('model.tflearn')

# Define a function to convert input text to a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# Define a function for the chatbot
def chat():
    print("The bot is ready to talk!! (Type 'quit' to exit)")
    while True:
        inp = input("\nYou: ")
        if inp.lower() == 'quit':
            break

        # Calculate the probability of each response
        results = model.predict([bag_of_words(inp, words)])

        # Find the index of the most probable response
        results_index = np.argmax(results)
        tag = labels[results_index]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
            print("Bot: " + random.choice(responses))

# Start the chatbot
chat()