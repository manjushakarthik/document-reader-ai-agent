import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
    else:
        text = ""
    return text

st.title("📄 Document Q&A Bot")

uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if not raw_text:
        st.error("Sorry, this file type is not supported or contains no extractable text.")
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(raw_text)

        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        model = OllamaLLM(model="llama3.2")
        template = """
        You are an expert in answering questions based on the following document excerpts:

        {reviews}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model

        st.success("Hurray!!! Document indexed. You can now ask questions!")

        question = st.text_input("Ask a question:")

        if question:
            with st.spinner("Thinking..."):
                relevant_chunks = retriever.invoke(question)
                answer = chain.invoke({"reviews": relevant_chunks, "question": question})
            st.markdown("### 🧠 Answer:")
            st.write(answer)
