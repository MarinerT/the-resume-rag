import streamlit as st
import time
import openai
from typing import Dict, List, Union
from langchain.retrievers.multi_query import MultiQueryRetriever
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def make_synchronous_openai_call(
    *,
    openai_api_key: str,
    model: str,
    temperature: float,
    messages: List[Dict[str, Union[str, Dict[str, str]]]],
    timeout_seconds: int,
):
    return openai.ChatCompletion.create(
        api_key=openai_api_key,
        model=model,
        messages=messages,
        top_p=1,
        n=1,
        max_tokens=st.secrets.openai.MAX_TOKENS,
        temperature=temperature,
        presence_penalty=0,
        frequency_penalty=0,
        logit_bias={},
        stream=False,
        request_timeout=timeout_seconds,
    )


def retrieve_resume_documents(
        llm,
        user_prompt,
        retriever) -> list:
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriever, llm=llm)
    unique_docs = retriever_from_llm.get_relevant_documents(query=user_prompt)
    return unique_docs


def generate_search_results(
    *,
    retriever,
    llm,
    user_prompt: str,
    timeout_seconds: int = 90,
) -> str:

    start_time = time.time()

    documents = retrieve_resume_documents(llm, user_prompt, retriever)

    messages = [
        {
            "role": "system",
            "content": (
                "You're an personal assistant tasked with helping recruiters find relevant experience from Todd's resume. "
                "Your task is to provide as many relevant documents as possible. "
                "Lastly, generating results swiftly should be prioritized over achieving perfection."
            ),
        },
        {
            "role": "user",
            "content": "I'll provide input as text of a list of Documents in content that follows '!!!. "
            "Each item in the list contains page_content and metadata."
            "Provide a brief summary of all the documents."
            "Give the section from the metadata and the related content."
            "Provide the information in short bullet points and provide the metadata with each document laid as such:"
            "if a word is between * and *, make the word appear bold."
            "*Summary*: "
            "*Section*: "
            "*Supporting Details*:"
            "Do not make stuff up. If a document has no valuable information, skip it."
            f"Here is the input !!!\n{str(documents)}",
        },
    ]
    start_time = time.time()

    openai_response = make_synchronous_openai_call(
        openai_api_key=st.secrets.openai.OPENAI_API_KEY,
        model=st.secrets.openai.OPENAI_MODEL,
        temperature=st.secrets.openai.temperature,
        messages=messages,
        timeout_seconds=timeout_seconds,
    )
    spent_time = time.time() - start_time
    print(f"Search took {spent_time} seconds")
    return openai_response["choices"][0]["message"]["content"]
