from ollama import chat 

messages = [
    {
        "role" : "system",
        "content" : "You are a python programming teacher"
    }
]

print("Marvellous Python Tutor") 
print("Type exit to stop")  

while True: 
    # Accept user input 
    print("-" * 40)
    question = input("\nYou : ").strip()

    # Stop chatbot 
    if question.lower() == "exit":
        print("Thank You for using Marvellous Bot") 
        break 

    # Ignore empty line 
    if not question:
        continue

    messages.append(
        {
            "role" : "user",
            "content" : question 
        }
    )

    response = chat(
        model = "llama3.2:1b",
        messages=messages   
    )

    answer = response.message.content

    print("\nMarvellous AI : ", answer) 

    messages.append({
        "role" : "assistant", 
        "content" : answer 
    })