import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

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
        
        # Use HuggingFace embeddings instead of Ollama
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
)
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        hf_pipeline = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-small",
            max_length=512,
            temperature=0.7,
            device=-1  # Use CPU
)
        model = HuggingFacePipeline(pipeline=hf_pipeline)
        
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
                st.markdown("### Answer:")
                st.write(answer)