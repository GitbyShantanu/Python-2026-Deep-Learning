from tensorflow.keras.preprocessing.text import Tokenizer 

sentences = [
    "food was good",
    "food was bad",
    "food was not good" 
]

tokenizer = Tokenizer() 

tokenizer.fit_on_texts(sentences) 

word_index = tokenizer.word_index
print(word_index)

for index, word in enumerate(word_index):
    print("Position", index, ":", word) 