import streamlit as st
from langchain.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
import pinecone

# Load resume
from toddbo.loader_utils import unzip, fetch_load_split
from toddbo.datastores import connect_to_chroma, connect_to_collection

unzip()

index_name = st.secrets.pinecone.index
OPENAI_API_KEY = st.secrets.openai.OPENAI_API_KEY


# Load Pinecone
@st.cache_resource
def load_pinecone(
    _documents, embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
):
    pinecone.init(api_key=st.secrets.pinecone.api_key)
    if index_name not in pinecone.list_indexes():
        return None
    docsearch = Pinecone.from_documents(
        _documents, embeddings, index_name=st.secrets.pinecone.index
    )
    return docsearch


def build_pinecone_retriever(search_type="mmr"):
    documents = fetch_load_split()
    vectordb = load_pinecone(documents)
    if vectordb is not None:
        retriever = vectordb.as_retriever(search_type=search_type)
        return retriever


# CHROMA
def generate_context(docsearch, kwargs):
    retriever = docsearch.as_retriever(**kwargs)
    return retriever


def build_chroma_retriever(
    prompt: str,
):
    client = connect_to_chroma(
        chroma_host=st.secrets.chroma.CHROMA_HOST,
        chroma_port=st.secrets.chroma.CHROMA_PORT,
    )
    retriever = generate_context(
        client,
        embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        collection_name=st.secrets.chroma.COLLECTION,
    )
    llm = ChatOpenAI(temperature=st.secrets.openai.temperature)
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm)

    unique_docs = retriever_from_llm.get_relevant_documents(query=prompt)
    return unique_docs
