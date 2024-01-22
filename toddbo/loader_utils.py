import zipfile as z
import tiktoken
import uuid
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st

def unzip() -> None:
    " unzips resume.zip and dumps into current directory as resume/"
    with z.ZipFile('./resume.zip', 'r') as zip_ref:
        zip_ref.extractall()

tokenizer = tiktoken.get_encoding('cl100k_base')
def tiktoken_len(text) -> int:
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def fetch_load_split(directory="resume/", chunk_size=128, chunk_overlap=64) -> list:
    loader = DirectoryLoader(directory)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
        )
    documents = loader.load_and_split(text_splitter=text_splitter)
    return documents

def load_to_pinecone(formatted_documents,namespace="v1", batch_size=100) -> None:
    pc = Pinecone(api_key=st.secrets.pinecone.api_key)
    index = pc.Index(st.secrets.pinecone.index)
    batch_limit = 100
    for i in range(0, len(formatted_documents), batch_size):
        index.upsert(vectors=formatted_documents[i:i+batch_size], namespace=namespace)


## CREATING THE VECTORS

def get_embeddings(documents, model_name='text-embedding-ada-002'):
    embed = OpenAIEmbeddings(model=model_name)
    texts = [document.page_content for document in documents]
    return embed.embed_documents(texts)

def get_section(source) -> str:
    section = source.split("/")[-2:]
    section[1] = section[1].split(".")[0]
    return f"{section[0].title()} - {section[1].replace('_', ' ').title()}"

def pull_source(document) -> str:
    return document.metadata['source']
    
def create_metadata(document) -> dict:
    source = pull_source(document)
    section = get_section(source)
    document.metadata.update({"section": section})
    return document.metadata

def format_to_json(documents) -> list:
    new_list = []
    embeddings = get_embeddings(documents)
    for i, document in enumerate(documents):
        temp = {"id": str(uuid.uuid4()),
                "values": embeddings[i],
                "metadata": create_metadata(document)
                }
        new_list.append(temp)
    return new_list
