import streamlit as st
from ollama import chat

st.title("Marvellous Python Tutor", icon=":material/smart_toy:")
st.caption("Python learning assistant powered by Llama 3.2:1b") 

SYSTEM_PROMPT = (
    "You are a expert Python programming trainer. " 
    "Explain concepts clearly with simple examples."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ] 

# Sidebar controls
with st.sidebar:
    st.subheader("Session Controls")

    if st.button("Reset Conversation"):
        st.session_state.messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
        st.rerun()

# Display previous conversation
for message in st.session_state.messages:
    if message["role"] == "system":
        continue

    # This is for 2nd msg and onwards, display user and assistant msgs before user input (for lateral chats) 
    with st.chat_message(message["role"]): # displays bubble avatar (user/assistant) and puts content in bubble before input
        st.write(message["content"]) 

# User input
user_input = st.chat_input("Ask a question...") # displays input box with placeholder text

if user_input:
    # Save and display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):  
        st.write(user_input) # displays user input in bubble 

    # Get response from Llama
    with st.chat_message("assistant"):
        try:
            with st.spinner("Llama is thinking..."):
                response = chat(
                    model="llama3.2:1b",
                    messages=st.session_state.messages
                )

                answer = response.message.content

            st.write(answer)

            # Save assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:
            st.error(f"Unable to connect to llama: {error}")
