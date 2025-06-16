import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
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
    if not raw_text.strip():
        st.error("Sorry, this file contains no extractable text.")
    else:
        with st.spinner("Processing document..."):
            # Split text into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(raw_text)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Build vector store
            vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            # Load text2text generation model (FLAN-T5)
            hf_pipeline = pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=200,
                temperature=0.7,
                device=-1
            )
            
            model = HuggingFacePipeline(pipeline=hf_pipeline)
            
            # Define prompt template
            template = """Answer the question based on the following context:

{reviews}

Question: {question}
Answer:"""
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | model

        st.success("✅ Document processed! You can now ask questions.")
        
        question = st.text_input("Ask a question:")
        if question:
            with st.spinner("Thinking..."):
                relevant_docs = retriever.invoke(question)
                # Only pass top 2-3 chunks to reduce prompt length
                top_chunks = relevant_docs[:3] if isinstance(relevant_docs, list) else [relevant_docs]
                context = "\n\n".join([doc.page_content for doc in top_chunks])
                # relevant_docs = retriever.invoke(question)
                # # Get only text from page_content
                # context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Run the prompt with context and question
                answer = chain.invoke({"reviews": context, "question": question})

                # Clean and display answer
                cleaned = answer.strip().replace("Human:", "").replace("Answer:", "").strip()
                st.markdown("### Answer:")
                st.write(cleaned)
