# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate

# st.set_page_config(page_title="Pizza Q&A Bot", layout="wide")
# st.title("🍕 Pizza Review AI Bot")

# # Upload section
# uploaded_file = st.file_uploader("Upload a PDF with pizza reviews", type="pdf")

# if uploaded_file:
#     # Extract text
#     reader = PdfReader(uploaded_file)
#     raw_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

#     # Split text
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#     chunks = splitter.split_text(raw_text)

#     # Embed & store
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#     vectorstore = Chroma.from_texts(chunks, embedding=embeddings)
#     retriever = vectorstore.as_retriever()

#     # LLM + Prompt
#     model = OllamaLLM(model="llama3.2")
#     template = """
#     You are an expert in answering questions about a pizza restaurant.

#     Here are some relevant reviews: {reviews}

#     Here is the question to answer: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#     chain = prompt | model

#     st.success("✅ Document indexed. You can now ask questions below.")

#     question = st.text_input("Ask a question about the reviews:")

#     if question:
#         with st.spinner("Thinking..."):
#             reviews = retriever.invoke(question)
#             result = chain.invoke({"reviews": reviews, "question": question})
#         st.markdown("### 🧠 Answer:")
#         st.write(result)
