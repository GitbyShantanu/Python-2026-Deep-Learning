import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Step 1 : Load the data

train_sentences = [
    "Food was good",
    "Food was bad", 
    "Food was excellent",
    "Food was terrible",
    "Service was good",
    "Service was bad",
    "Service was excellent",
    "Service was terrible",
    "Ambience was good",
    "Ambience was bad",
    "Ambience was excellent",
    "Ambience was terrible"
]

train_labels = [
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0
]

# Step 2 : Tokenization
tokenizer = Tokenizer(oov_token="<OOV>") 

tokenizer.fit_on_texts(train_sentences) 

# Step 3 : Convert training data into sequence 

train_sequence = tokenizer.texts_to_sequences(train_sentences) 

print("Training sequences : ") 

for sentence, sequence in zip(train_sentences, train_sequence):
    print(sentence, " -> ", sequence)
