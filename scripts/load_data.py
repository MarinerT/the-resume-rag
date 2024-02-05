import streamlit as st
from toddbo import connect_to_chroma, unzip, fetch_load_split, connect_to_collection, upload_to_collection

# Load data to Chroma

if __name__ == '__main__':
    print('Loading data to Chroma...')

    client = connect_to_chroma(chroma_host=st.secrets.chroma.CHROMA_HOST,
                               chroma_port=st.secrets.chroma.CHROMA_PORT
                               )
    unzip()
    documents = fetch_load_split()
    collection = connect_to_collection(client, st.secrets.chroma.COLLECTION)
    upload_to_collection(documents, collection)
