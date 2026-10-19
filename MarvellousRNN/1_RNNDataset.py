sentences = [
    "food was good",
    "food was bad",
    "food was not good" 
]

labels = [1,0,0] 

for sentence, label in zip(sentences, labels):
    sentiment = "Positive" if label == 1 else "Negative" 
    print("Sentense : ", sentence) 
    print("Label : ", label) 
    print("Meaning : ", sentiment) 
    print("--------------------------------------------------")
