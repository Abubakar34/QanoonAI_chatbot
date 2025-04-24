import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
faiss_vector_store = FAISS.load_local(
    "./faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = faiss_vector_store.as_retriever(kwargs={"search_kwargs": {"k": 5}})

llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant and your name is 'Jolly LLB' that answers questions about law in saimple and understandable language based on the given context only. if the context does not have enough information then say I'don;t have information abouth this question. Context is: {context}.",
        ),
        ("human", "{input}"),
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(
    retriever=retriever, combine_docs_chain=document_chain
)


# App title and description
st.title("Qanoon.AI")
st.write(
    "Ask questions about the criminal offences and their punishments mentioned in PPC(Pakistan Penal Code)!"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask your question...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing constitution..."):
            response = retriever_chain.invoke({"input": user_query})
            answer = response["answer"]
            answer = answer[(answer.find("</think>") + 8) :]

        st.markdown(answer)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
