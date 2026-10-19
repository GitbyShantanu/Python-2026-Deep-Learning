from ctypes.wintypes import PINT
from traceback import print_tb

from ollama import chat

# Select the local LLM
Model = "llama3.2:1b" 

# Define prompts  
prompts = [
    "Explain recursion",

    "Explain recursion in python in 3 simple points",

    """You are experienced python teacher.
    explain recursion for beginner. 
    Give one simple example.""",

    """Convert programming languages to Uppercase. 
    Example 1 : 
    Input : Python 
    Output : PYTHON

    Example 2 : 
    Input : Java
    Output : JAVA

    Example 3 : 
    Input : javascript 
    Output : """

]

# Execute every prompt 
for index, prompt in enumerate(prompts, start=1):
    print("="*30)
    print("Prompt Number : ", index) 
    print("Input Prompt : ", prompt) 

    response = chat(
        model = Model,
        messages = [
            {
                "role" : "user",
                "content" : prompt
            }
        ]
    )

    print("Response from LLM : ")
    print(response.message.content) 

    