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

# Step 4 : Apply Padding    
max_length = 4

X_train = pad_sequences(
    train_sequence,
    maxlen=max_length,
    padding='pre'
)

Y_train = np.array(train_labels) 

print("Padded training data") 
print(X_train) 

print("Training Labels : ")
print(Y_train)

# Step 5: Calculate vocabulary size 
vocab_size = len(tokenizer.word_index) + 1 

print("Vocab size : ", vocab_size) 

# Step 6 : Build RNN Model 

model = Sequential() 

model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=8,
        input_length=max_length 
    )
)

model.add(
    SimpleRNN(
        units=8,
        activation="tanh",
    )
)

model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)

# Step 7 : Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"] 
)

# Step 8 : Display Model
model.build(input_shape=(None, max_length)) 

print("Model Architecture") 
model.summary() 

# Step 9 : Train the model
history = model.fit(
    X_train, 
    Y_train,
    epochs = 50, 
    verbose = 1
)

print("Model training completed") 

# Step 10: Create unseen data 
test_sentences = [
    "Service was amazing",
    "Service was horrible",
    "Experience was excellent",
    "Experience was terrible"
]

# Step 11 : Convert text to sequence
test_sequences = tokenizer.texts_to_sequences(test_sentences)

X_test = pad_sequences(
    test_sequences,
    maxlen=max_length,
    padding="pre" 
)

# Step 12 : Predict the sentiment 

for text, sequence, padded in zip(test_sentences, test_sequences, X_test):
    input_data = np.array([padded]) 

    prediction = model.predict(input_data, verbose=0) 

    probability = float(prediction[0][0]) 

    print("Sentence : ", text) 
    print("Sequence : ", sequence) 
    print("Padded sequence : ", padded)
    print("Prediction : ", prediction) 

    if probability >= 0.5:
        print("Sentiment : Positive") 
    else:
        print("Sentiment : Negative")