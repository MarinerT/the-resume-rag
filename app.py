import os
import streamlit as st
import streamlit_authenticator as stauth
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
# from toddbo.chain import make_synchronous_openai_call, retrieve_resume_documents
# from toddbo.retriever import build_retriever

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
        os.environ['OPENAI_API_KEY'] = st.secrets.openai.OPENAI_API_KEY
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
        st.title("In order to use the chatbot, you need to be signed in.")
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
        st.title("In order to use the chatbot, you need to be signed in.")

# StreamHandler to intercept streaming output from the LLM.
# This makes it appear that the Language Model is "typing"
# in realtime.
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


@st.cache_resource
def create_chain(system_prompt):
    # A stream handler to direct streaming output on the chat screen.
    # This will need to be handled somewhat differently.
    # But it demonstrates what potential it carries.
    # stream_handler = StreamHandler(st.empty())

    # Callback manager is a way to intercept streaming output from the
    # LLM and take some action on it. Here we are giving it our custom
    # stream handler to make it appear that the LLM is typing the
    # responses in real-time.
    # callback_manager = CallbackManager([stream_handler])

    # (repo_id, model_file_name) = ("TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    #                               "mistral-7b-instruct-v0.1.Q4_0.gguf")

    # model_path = hf_hub_download(repo_id=repo_id,
    #                              filename=model_file_name,
    #                              repo_type="model")

    # initialize LlamaCpp LLM model
    # n_gpu_layers, n_batch, and n_ctx are for GPU support.
    # When not set, CPU will be used.
    # set 1 for Mac m2, and higher numbers based on your GPU support
    # llm = LlamaCpp(
    #         model_path=model_path,
    #         temperature=0,
    #         max_tokens=512,
    #         top_p=1,
    #         # callback_manager=callback_manager,
    #         # n_gpu_layers=1,
    #         # n_batch=512,
    #         # n_ctx=4096,
    #         stop=["[INST]"],
    #         verbose=False,
    #         streaming=True,
    #         )
    llm = ChatOpenAI(temperature=st.secrets.openai.temperature, model_name=st.secrets.openai.generation_model)

    # Template you will use to structure your user input before converting
    # into a prompt. Here, my template first injects the personality I wish to
    # give to the LLM before in the form of system_prompt pushing the actual
    # prompt from the user. Note that this chatbot doesn't have any memory of
    # the conversation. So we will inject the system prompt for each message.
    # template = """
    # <s>[INST]{}[/INST]</s>

    # [INST]{}[/INST]
    # """.format(system_prompt, "{question}")

    template = """ {} {}""".format(system_prompt, "{question}")

    # We create a prompt from the template so we can use it with Langchain
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # We create an llm chain with our LLM and prompt
    # llm_chain = LLMChain(prompt=prompt, llm=llm) # Legacy
    llm_chain = prompt | llm  # LCEL

    return llm_chain


# Create a header element
st.header("Chat with Todd's Resume Assistant!")

# This sets the LLM's personality for each prompt.
# The initial personality provided is basic.
# Try something interesting and notice how the LLM responses are affected.
# system_prompt = st.text_area(
#     label="System Prompt",
#     value="You are a helpful AI assistant who answers questions in short sentences.",
#     key="system_prompt")

system_prompt = (
                "You're an assistant tasked with helping users by finding relevant documents. "
                "Your task is to provide as many relevant documents as possible "
                "while providing main key points on why each document is relevant as well provide its source. "
                "Lastly, generating results swiftly should be prioritized over achieving perfection."
                "Separate each entry by line."
            )

# Create LLM chain to use for our chatbot.
llm_chain = create_chain(system_prompt)
# llm = ChatOpenAI(temperature=st.secrets.openai.temperature, model_name=st.secrets.openai.generation_model)


# Build the retriever
# retriever = build_retriever()

# We store the conversation in the session state.
# This will be used to render the chat conversation.
# We initialize it with the first message we want to be greeted with.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

# We loop through each message in the session state and render it as
# a chat message.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# We take questions/instructions from the chat input to pass to the LLM
if user_prompt := st.chat_input("Your message here", key="user_input"):

    # Add our input to the session state
    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )

    # Add our input to the chat window
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Pass our input to the LLM chain and capture the final responses.
    # It is worth noting that the Stream Handler is already receiving the
    # streaming response as the llm is generating. We get our response
    # here once the LLM has finished generating the complete response.
    if authentication_status:
        response = llm_chain.invoke({"question": user_prompt})
        #get relevant documents 
        # documents = retrieve_resume_documents(llm, user_prompt, retriever)
    
        # messages = [
        #     {
        #         "role": "system",
        #         "content": (
        #             "You're an assistant tasked with helping users by finding relevant documents. "
        #             "Your task is to provide as many relevant documents as possible "
        #             "while providing main key points on why each document is relevant as well provide its source. "
        #             "Lastly, generating results swiftly should be prioritized over achieving perfection."
        #         ),
        #     },
        #     {
        #         "role": "user",
        #         "content": "I'll provide input as text of a list of Documents in content that follows '!!!. "
        #         "Each item in the list contains page_content and metadata."
        #         "provide key facts per page and give the section from the metadata." 
        #         " Provide the information in short bullet points and provide the metadata with each document laid as such:"
        #         "if a word is between * and *, make the word appear bold."
        #         "*Key Facts per Page*: "
        #         "*Section*: "
        #         "Do not make stuff up. If a document has no valuable information, skip it."
        #         f"Here is the input !!!\n{str(documents)}",
        #     },
        # ]
    
        # response = make_synchronous_openai_call(
        #     openai_api_key=st.secrets.OPENAI_API_KEY,
        #     model=st.secrets.openai.OPENAI_MODEL,
        #     temperature=st.secrets.openai.temperature,
        #     messages=messages,
        #     timeout_seconds=timeout_seconds,
        # )

        # Add the response to the session state
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response.content)
    else:
        st.markdown("please enter the password.")
