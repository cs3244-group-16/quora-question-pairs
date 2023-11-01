#Read in the training set
import pandas as pd
Train = pd.read_csv("/Users/hongruih/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Year 3 Semester One/CS3244/Project/Test Codes/Train_subset.csv")
Train = Train.dropna()
#Train = Train.loc[105769:105790,]
#Pre-Processing

import string

###This code is from ChatGPT, used to remove punctuations
def remove_punctuation(text):
    # Create a translation table to remove punctuation characters
    translator = str.maketrans('', '', string.punctuation)
    
    # Use the translation table to remove punctuation
    text_without_punctuation = text.translate(translator)
    
    return text_without_punctuation


#Extract the first questions and store it in a variable
Q1 = Train["question1"].copy()
#Extract the second questions and store it in a variable
Q2 = Train["question2"].copy()


##Lowercasing all question pairs and Removing Punctuation

for i in range(len(Q1)):
    #print(Q1[i])
    Q1.iloc[i] = remove_punctuation(Q1.iloc[i].lower())
    Q2.iloc[i] = remove_punctuation(Q2.iloc[i].lower())

#Use a pre-trained tokenizer model from the spaCy library to tokenize a question

import spacy

#From the spacy library: Import a pre-trained tokenizer
Tokenizer = spacy.load("en_core_web_sm")

#Extract the tokens from each of the questions

Q1_Tokens = Q1.copy()
Q2_Tokens = Q2.copy()

for i in range(len(Q1_Tokens)):
    #print(i)
    #print(Q1_Tokens.iloc[i])
    #print(Q2_Tokens.iloc[i])
    Q1_Tokens.iloc[i] = [token.text for token in Tokenizer(Q1_Tokens.iloc[i])]
    Q2_Tokens.iloc[i] = [token.text for token in Tokenizer(Q2_Tokens.iloc[i])]

#Use Word2Vec to embed each question individually questions

from gensim.models import Word2Vec

Q1_Embedded = Q1_Tokens.copy()
Q2_Embedded = Q2_Tokens.copy()
#For each question: Embed it using Word2Vec
for i in range(len(Q1_Embedded)):
    Q1_Embedded.iloc[i] = Word2Vec([Q1_Embedded.iloc[i]],vector_size=100, window=5, min_count=1, sg=0)
    Q2_Embedded.iloc[i] = Word2Vec([Q2_Embedded.iloc[i]],vector_size=100, window=5, min_count=1, sg=0)

#Aggregate each question to get a vector representation of each question, using the average method

#For each question pair
for i in range(len(Q1_Embedded)):
    
    #For each question: Extract the vector representation for each word and average them
    
    #Create a tracker that will store the running sum of the word's vectors in the first question in this pair
    Sum_Q1 = 0
    #For each word in the first question in this pair
    for word in Q1_Tokens.iloc[i]:
        #Extract the vector and add it to the running sum
        Sum_Q1 = Sum_Q1 + Q1_Embedded.iloc[i].wv[word]
    #Find the average of these sums ~ This is the vector representation of the first question in this pair
    Sum_Q1 = Sum_Q1/len(Q1_Tokens)
    #Overwrite the list with this new vector
    Q1_Embedded.iloc[i] = Sum_Q1
    #Create a tracker that will store the running sum of the word's vectors in the second question in this pair
    Sum_Q2 = 0
    #For each word in the second question in this pair
    for word in Q2_Tokens.iloc[i]:
        #Extract the vector and add it to the running sum
        Sum_Q2 = Sum_Q2 + Q2_Embedded.iloc[i].wv[word]
    #Find the average of these sums ~ This is the vector representation of the second question in this pair
    Sum_Q2 = Sum_Q2/len(Q2_Tokens)
    #Overwrite the list wih this vector
    Q2_Embedded.iloc[i] = Sum_Q2


#Initialize the Siamese network architecture

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Identify the length of the vectors used in the embeddings
embedding_dim = 100

#Defining the shared network ~ This initializes what network will be used to process our inputs
#The word "shared" is used because both networks will be using the exact architecture
#In this case: It uses 2 dense layers with relu activation functions ~ The first layer takes in the inputs
shared_network = keras.Sequential([
    #First layer
    layers.Dense(128, activation='relu', input_shape=(embedding_dim,)),
    #Second layer
    layers.Dense(128, activation='relu')
])

#Define the left and right inputs for the question pair ~ Initializing how long the vectors that represent each question is
left_input = layers.Input(shape=(embedding_dim,))
right_input = layers.Input(shape=(embedding_dim,))

# Encode the question pair using the shared network ~ Indicate that we will input the questions both question in the pair into the shared network
encoded_left = shared_network(left_input)
encoded_right = shared_network(right_input)

# Calculate the Euclidean distance between the encodings ~ I.e: We will use this distance function to determine the similarity between the outputs
distance = layers.Lambda(lambda x: tf.norm(x[0] - x[1], axis=1, keepdims=True))([encoded_left, encoded_right])

# We have initialized the model. Now, just create the Siamese model
siamese_model = keras.Model(inputs=[left_input, right_input], outputs=distance)


#Compile the model ~ Specifies how the training should be done etc.....
siamese_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training of the model ~ We will use only 1 question for this experiment

#Inputs has to be an array of a list of lists -> Convert from array of list to a list 

Q1_Inputs = []

for i in range(len(Q1_Embedded)):
    Q1_Inputs.append(Q1_Embedded.iloc[i].tolist())

Q1_Inputs = np.array(Q1_Inputs)

Q2_Inputs = []

for i in range(len(Q2_Embedded)):
    Q2_Inputs.append(Q2_Embedded.iloc[i].tolist())

Q2_Inputs = np.array(Q2_Inputs)


#Training of the model

siamese_model.fit(
    [Q1_Inputs, Q2_Inputs],  # Your question embeddings
    np.array(Train["is_duplicate"]),  # Similarity labels (0 for dissimilar, 1 for similar)
    batch_size=32,
    epochs=10,
    validation_split=0  # You can adjust the validation split
)


#Predicting the above test points:
Results = (siamese_model.predict([Q1_Inputs, Q2_Inputs]))
Prediction = []
for i in range(len(Results)):
    if Results[i]>0.5:
        Prediction.append(1)
    else:
        Prediction.append(0)

#Calculating training accuracy:

Score = 0

for i in range(len(Prediction)):
    if Prediction[i] == Train["is_duplicate"].tolist()[i]:
        Score = Score + 1

print("Accuracy is")
print(Score/len(Prediction))


    


