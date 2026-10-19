from tensorflow.keras.preprocessing.text import Tokenizer 

sentences = [
    "food was good",
    "food was bad",
    "food was not good" 
]

tokenizer = Tokenizer() 

tokenizer.fit_on_texts(sentences) 

sequences = tokenizer.texts_to_sequences(sentences)

for sentence, sequance in zip(sentences, sequences):
    print("Sentence : ", sentence) 
    print("Sequance : ", sequance) 
    print("------------------------------------")