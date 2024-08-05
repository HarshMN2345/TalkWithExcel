import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

# Streamlit interface
st.title("Excel Data Question Answering System")

# API key inputs
groq_api = st.text_input("Enter your Groq API key:", type="password")
google_api = st.text_input("Enter your Google API key:", type="password")

# File uploader
uploaded_files = st.file_uploader("Upload 5 Excel files", type="xlsx", accept_multiple_files=True)

if groq_api and google_api:
    # Set the API keys
    os.environ["GOOGLE_API_KEY"] = google_api

    # Initialize language model and embeddings
    llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Function to process a single Excel file
    def process_excel_file(file):
        documents = []
        xls = pd.ExcelFile(file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file, sheet_name=sheet_name)
            for _, row in df.iterrows():
                content = f"File: {file.name}, Sheet: {sheet_name}\n" + "\n".join([f"{col}: {row[col]}" for col in df.columns])
                document = Document(page_content=content, metadata={"source": file.name, "sheet": sheet_name})
                documents.append(document)
        return documents

    # Function to load Excel files and create embeddings
    @st.cache_resource
    def load_and_embed_excel_files(uploaded_files):
        all_documents = []
        for file in uploaded_files:
            all_documents.extend(process_excel_file(file))
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(all_documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        return vector_store

    if uploaded_files and len(uploaded_files) == 5:
        with st.spinner("Processing Excel files and creating embeddings..."):
            vector_store = load_and_embed_excel_files(uploaded_files)
        
        st.success("Files processed successfully!")
        
        retriever = vector_store.as_retriever()
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}""")
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        user_question = st.text_input("Ask a question about the Excel data:")
        if user_question:
            with st.spinner("Generating answer..."):
                response = retrieval_chain.invoke({"input": user_question})
            st.write("Answer:", response["answer"])
    elif uploaded_files:
        st.warning("Please upload exactly 5 Excel files.")
    else:
        st.info("Please upload 5 Excel files to start.")
else:
    st.warning("Please enter both Groq and Google API keys to proceed.")