import os
import gradio as gr
from groq import Groq

# Get Groq API key from environment variable
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set.")

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)


# Chat function
def chat(message, history):

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI study assistant. "
                "Explain Java, Python, C and ECE topics in simple language. "
                "Give clear, beginner-friendly explanations and examples."
            )
        }
    ]

    # Add previous conversation
    if history:
        messages.extend(history)

    # Add current user message
    messages.append({
        "role": "user",
        "content": message
    })

    # Call Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages
    )

    return response.choices[0].message.content


# Create Gradio chatbot
demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    title="🤖 My AI Study Chatbot",
    description="Ask me anything about Java, Python, C, ECE and engineering!"
)


# Launch application
demo.launch()
