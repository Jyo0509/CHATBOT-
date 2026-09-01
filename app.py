import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="My AI Study Chatbot",
    page_icon="🤖"
)

st.title("🤖 My AI Study Chatbot")
st.write("Ask me anything about Java, Python, C, ECE and engineering!")

# Get Groq API key from Streamlit Secrets
api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI study assistant. "
                "Explain Java, Python, C and ECE topics in simple language. "
                "Give clear, beginner-friendly explanations and examples."
            )
        }
    ]

# Display previous messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    # Get response from Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=st.session_state.messages
    )

    answer = response.choices[0].message.content

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.write(answer)


