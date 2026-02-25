import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnablePassthrough

# API Key handling
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is missing. Please add it to your .env file.")
    st.stop()
os.environ["GROQ_API_KEY"] = api_key

# Initialize LLM
llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context provided.
    you need to provided the related image or diagrams for the concepts.
    please given the response accurately and in a concise manner.
    context:{context}
    question:{question}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
        # Loads all PDFs in the "research_paper" directory
        st.session_state.loader = PyPDFDirectoryLoader("research_paper")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectorstore = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.session_state.vectors = True

st.title("RAG DOCUMENTS WITH GROQ")

# User input
user_input = st.text_input("Ask a question about the research paper")

if st.button("Vector Embeddings"):
    with st.spinner("Creating vector embeddings..."):
        create_vector_embeddings()
        st.success("Vector store is ready!")

import time
if user_input:
    if "vectorstore" not in st.session_state:
        st.warning("Please create vector embeddings first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever()
        
        # LCEL Chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        start = time.process_time()
        response = rag_chain.invoke(user_input)
        end = time.process_time()
        
        # Documents for expander
        context_docs = retriever.invoke(user_input)
        
        st.write(f"Response (Time taken: {end-start:.2f}s):")
        st.write(response)

        # Streamlit expander
        with st.expander("Documents similarity search"):
            for i, doc in enumerate(context_docs):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write('--------------------')