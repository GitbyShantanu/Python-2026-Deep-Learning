words = ["food", "was", "not", "good"] 

hidden_state = "empty memory" 

print("Input tokens : ", words) 

print("Initial Hidden state : ", hidden_state) 

for index, word in enumerate(words):
    print("TimeStep : ", index+1)
    print("Current word : ", word) 
    print("Hidden state : ", hidden_state) 

    hidden_state = "Memory after reading : " + " ".join(words[: index+1]) + "" 
    print("Updated memory : ", hidden_state) 

    print("-" *30) 
