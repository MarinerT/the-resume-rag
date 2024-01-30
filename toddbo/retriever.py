from langchain.vectorstores import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
import pinecone

# Load resume
from toddbo.loader_utils import unzip, fetch_load_split

unzip()

index_name = st.secrets.pinecone.index
OPENAI_API_KEY = st.secrets.openai.api_key

# Load Pinecone
@st.cache_resource
def load_pinecone(_documents, embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)):
    pinecone.init(api_key=st.secrets.pinecone.api_key)
    if index_name not in pinecone.list_indexes():
        return None
    docsearch = Pinecone.from_documents(_documents, embeddings, index_name=st.secrets.pinecone.index)
    return docsearch

def retrieve_resume_records(search_type="mmr"):
    documents = fetch_load_split()
    vectordb = load_pinecone(documents)
    if vectordb is not None:
        retriever = vectordb.as_retriever(search_type=search_type)
        return retriever
    
