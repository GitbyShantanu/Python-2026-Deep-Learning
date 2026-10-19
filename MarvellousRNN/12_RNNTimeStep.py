sentence = "Food was not good" 

words = sentence.split() 

print("Actual sentance is : ", sentence) 

for index, word in enumerate(words):
    print("TimeStep : ", index+1, " : ", word) 