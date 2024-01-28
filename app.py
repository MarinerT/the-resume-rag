import os, pickle
import logging
import streamlit as st
import streamlit_authenticator as stauth
from toddbo.openai_constants import (
    MAX_TOKENS,
    GPT_3_5_TURBO_MODEL,
    GPT_3_5_TURBO_0301_MODEL,
    GPT_3_5_TURBO_0613_MODEL,
    GPT_3_5_TURBO_1106_MODEL,
    GPT_3_5_TURBO_16K_MODEL,
    GPT_3_5_TURBO_16K_0613_MODEL,
    GPT_4_MODEL,
    GPT_4_0314_MODEL,
    GPT_4_0613_MODEL,
    GPT_4_1106_PREVIEW_MODEL,
    GPT_4_32K_MODEL,
    GPT_4_32K_0314_MODEL,
    GPT_4_32K_0613_MODEL,
)

"OPENAI_API_KEY" = st.secrets.openai.OPENAI_API_KEY



st.set_page_config(page_title="Todd's ResumeBot", layout="wide")

st.title("Todd's Resume Bot")
st.header("Ask my resume questions!")
# --------------------------------------------------------------------------------------- #
#                                Sidebar & Authentication
with st.sidebar:
    st.subheader("Coming Soon!")
    st.caption("Copy/Paste your requirements and get a comparison to my resume using the multi-query fusion approach!")

    authenticator = stauth.Authenticate(
        dict(st.secrets['credentials']),
        st.secrets.cookie.name,
        st.secrets.cookie.key,
        st.secrets.cookie.expiry_days,
        st.secrets['preauthorized']
    )
    name, authentication_status, username = authenticator.login("Login", "sidebar")

    if st.session_state["authentication_status"]:
        authenticator.logout("Logout", "sidebar")
        st.write(f'Welcome *{st.session_state["name"]}*')
        st.title('Some content')
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')


selected_custom_name = "Todd's Resume"

if authentication_status:
    os.environ['OPENAI_API_KEY'] = st.secrets.openai.openai_api_key
else:
    st.title("In order to use the chatbot, you need to be signed in.")

# --------------------------------------------------------------------------------------- #
#                                  The Retriever

from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Load resume
from toddbo.loader_utils import unzip, fetch_load_split

unzip()

# load index from pinecone
@st.cache_resource
def load_pinecone(_documents, embeddings=OpenAIEmbeddings()):
    pinecone.init(api_key=st.secrets.pinecone.api_key)
    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=st.secrets.pinecone.index,
            metric='cosine',
            dimension=1536  
            )
    docsearch = Pinecone.from_documents(_documents, embeddings, index_name=st.secrets.pinecone.index)
    return docsearch

# # #--------------------------
# # # Create Chain

from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

def retrieve_resume_records(prompt):
    documents = fetch_load_split()
    vectordb = load_pinecone(documents)
    retriever = vectordb.as_retriever(search_type="mmr")

    llm = ChatOpenAI(temperature=st.secrets.openai.temperature, model_name=st.secrets.openai.generation_model)
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    unique_docs = retriever_from_llm.get_relevant_documents(query=prompt)
    return unique_docs

result = retrieve_resume_records(prompt="Where did Todd work in 2021?")
st.write(result)

# # #--------------------------

# def generate_search_results(
#     *,
#     logger: logging.Logger,
#     openai_api_key: str,
#     prompt: str,
#     timeout_seconds: int,
# ) -> str:
    
#     start_time = time.time()
    
#     documents = retrieve_resume_records(prompt, retriever)
    
#     messages = [
#         {
#             "role": "system",
#             "content": (
#                 "You're an assistant tasked with helping Slack users by finding relevant documents. "
#                 "Your task is to provide as many relevant documents as possible "
#                 "while providing main key points on why each document is relevant as well provide its source. "
#                 "Lastly, generating results swiftly should be prioritized over achieving perfection."
#             ),
#         },
#         {
#             "role": "user",
#             "content": "I'll provide input as text of a list of Documents in content that follows '!!!. "
#             "Each item in the list contains page_content and metadata."
#             "provide key facts and figures from page_content." 
#             " Provide the information in short bullet points and provide the metadata with each document laid as such:"
#             "if a word is between * and *, make the word appear bold."
#             "*Key Facts per Page*: "
#             "*Page*: "
#             "*Source*: "
#             "Do not make stuff up. If a document has no valuable information, skip it."
#             f"Here is the input !!!\n{str(documents)}",
#         },
#     ]
#     start_time = time.time()
    

#     openai_response = make_synchronous_openai_call(
#         openai_api_key=openai_api_key,
#         model=context["OPENAI_MODEL"],
#         temperature=context["OPENAI_TEMPERATURE"],
#         messages=messages,
#         user=context.actor_user_id,
#         openai_api_type=context["OPENAI_API_TYPE"],
#         openai_api_base=context["OPENAI_API_BASE"],
#         openai_api_version=context["OPENAI_API_VERSION"],
#         openai_deployment_id=context["OPENAI_DEPLOYMENT_ID"],
#         timeout_seconds=timeout_seconds,
#     )
#     spent_time = time.time() - start_time
#     logger.debug(f"Search took {spent_time} seconds")
#     return openai_response["choices"][0]["message"]["content"]

# # Display a chat window for our app
# # Initialize Streamlit chat UI

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Ask your questions from "f'{selected_custom_name}'"?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Load vectorstore using pickle
#     pickle_folder = "pickle"
#     pickle_file_path = os.path.join(pickle_folder, f"{selected_custom_name}.pkl")
#     if os.path.exists(pickle_file_path):
#         with open(pickle_file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#         retriever = vectorstore.as_retriever()
#         qa.retriever = retriever

#     result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})
    
#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = result["answer"]
#         message_placeholder.markdown(full_response + "|")
#     message_placeholder.markdown(full_response)
    
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
