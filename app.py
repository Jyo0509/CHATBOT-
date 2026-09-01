import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# PAGE SETTINGS
# -------------------------------

st.set_page_config(
    page_title="AI Study Chatbot",
    page_icon="🤖"
)

st.title("🤖 AI Study Chatbot")
st.write("Ask questions from your uploaded study material.")


# -------------------------------
# GROQ
# -------------------------------

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)


# -------------------------------
# PDF TEXT EXTRACTION
# -------------------------------

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -------------------------------
# CREATE CHUNKS
# -------------------------------

def create_chunks(text, chunk_size=800):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# -------------------------------
# RAG RETRIEVAL
# -------------------------------

def retrieve_chunks(question, chunks, top_k=3):

    vectorizer = TfidfVectorizer()

    documents = chunks + [question]

    vectors = vectorizer.fit_transform(documents)

    similarities = cosine_similarity(
        vectors[-1],
        vectors[:-1]
    )[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append(chunks[index])

    return results


# -------------------------------
# UPLOAD PDF
# -------------------------------

pdf_file = st.file_uploader(
    "📚 Upload your study PDF",
    type=["pdf"]
)


if pdf_file:

    text = extract_text(pdf_file)

    chunks = create_chunks(text)

    st.success(
        f"PDF loaded successfully! Created {len(chunks)} text chunks."
    )


    # -------------------------------
    # CHAT HISTORY
    # -------------------------------

    if "messages" not in st.session_state:

        st.session_state.messages = []


    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


    # -------------------------------
    # USER QUESTION
    # -------------------------------

    question = st.chat_input(
        "Ask something from your PDF..."
    )


    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.write(question)


        # -------------------------------
        # RETRIEVE RELEVANT CONTENT
        # -------------------------------

        relevant_chunks = retrieve_chunks(
            question,
            chunks,
            top_k=3
        )

        context = "\n\n".join(relevant_chunks)


        # -------------------------------
        # SEND TO GROQ
        # -------------------------------

        prompt = f"""
You are an AI study assistant.

Answer the user's question using the study material provided below.

If the answer is not present in the study material,
say that the information is not available in the uploaded document.

Explain in simple beginner-friendly language.

STUDY MATERIAL:
{context}

USER QUESTION:
{question}
"""


        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI study assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )


        answer = response.choices[0].message.content


        # -------------------------------
        # DISPLAY ANSWER
        # -------------------------------

        with st.chat_message("assistant"):

            st.write(answer)


        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info("👆 Upload a PDF to start chatting.")

