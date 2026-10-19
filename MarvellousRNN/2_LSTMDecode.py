# Step 1 : Import Required libraries

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

#########################################################
# Step 2 : Configuration of values
#########################################################

VOCAB_SIZE = 10000      # consider most frequent 10k unique words
MAX_LENGTH = 200        # consider maximum 200 words in review 


#########################################################
# Step 3 : Load the IMDB (International Movie DataBase) dataset 
#########################################################

print("-"*50)
print("Movie Review Sentiment Analysis Using LSTM")
print("-"*50) 

print("Loading the dataset...") 

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(number_words = VOCAB_SIZE) 

print("IMDB Dataset Loaded successfully") 

print("Number of training reviews : ", len(X_train)) 
print("Number of testing reviews : ", len(X_test))  


#########################################################
# X_train      Reviews used for training
# Y_train      Actual sentiments of training
# X_test       Reviews used for testing 
# Y_test       Actual sentiments of testing

# Sentiments : 
# 0 -> Negative Sentiment 
# 1 -> Positive Sentiment  cls

#########################################################


#########################################################
# Step 4 : Load the word dictionary 
#########################################################

word_index = imdb.get_word_index()

# Dictionary contains mapping of word and its corresponding number 
# Drishyam is good movie  -> (20 56 78 43)
# 20 -> Drishyam
# 56 -> is
# 78 -> good
# 43 -> movie


#########################################################
# Step 5 : Create reverse Dictionary 
#########################################################

reverse_word_index = {} 

for word, index in word_index.items():
    reverse_word_index[index+3] = word 


#########################################################
# Step 6 : Function to decode the review (numbe to word) 
#########################################################

def DecodeReview(encoded_review):
    words = [] 

    for number in encoded_review:
        if number >= 3: # ignore the first 3
            word = reverse_word_index.get(number, "?") # 
            words.append(word) 

    return " ".join(words) # join the list of words 

#########################################################
# Step 7 : Display Sample Reviews 
#########################################################

print("-"*50) 
print("-------------Sample Reviews---------------") 
print("-"*50) 


for i in range(5): 
    review = DecodeReview(X_train[i]) 

    print("-"*50) 
    
    print("Review Number : ", i+1) 
    print("Review : ") 
    print(review) 

    print("-"*50) 
    
    if Y_train[i] == 1:
        print("Sentiment :  POSITIVE") 
    else : 
        print("Sentiment : NEGATIVE")

    
