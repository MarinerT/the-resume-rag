# for more info https://docs.trychroma.com/usage-guide
import os
import uuid
from typing import List, Dict, Tuple
import numpy as np
import chromadb
import tiktoken
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from langchain.schema import Document
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

openai_api_key = st.secrets.openai.OPENAI_API_KEY


def connect_to_chroma(
    chroma_host: str = None, chroma_port: int = None, local: bool = True
):
    """
    DESCRIPTION
    -----------

    creates a python client for chromadb instance

    INPUT
    -----

    nothing unless you want to specify
     - chroma_host: the host url as a string
     - chroma_port: the port for the fastapi connection as an integer
     - local: if set to True [default] as boolean, connects to a local host

    OUTPUT
    ------

    chromadb.PersistentClient if local otherwise chromadb.HttpClient

    EXAMPLES
    --------

    # Instantiates a chromadb.PersistentClient - local version
    client = galdre.connect_to_chroma()

    # Instantiates a chromadb.HttpClient to baseurl localhost:8000
    client = galdre.connect_to_chroma(local=False)

    # chromadb.HttpClient to a remote server
    client = galdre.connect_to_chroma(
        chroma_host="someurl.com",
        chroma_port=80
        )

    """
    if local:
        DIR = os.path.dirname(os.path.abspath(__file__))
        DB_PATH = os.path.join(DIR, "data", "db")
        return chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

    chroma_host = "localhost" if chroma_host is None else chroma_host
    chroma_port = 8000 if chroma_port is None else chroma_port
    return chromadb.HttpClient(host=chroma_host, port=chroma_port)


def connect_to_collection(
    client: chromadb.HttpClient,
    collection: str,
    metadata: dict = {"hnsw:space": "cosine"},
    embedding_function=None,
    overwrite: bool = False,
) -> chromadb.Collection:
    """
    DESCRIPTION
    -----------

    creates a connection to the collection

    INPUT
    -----

     - client: an client instance of PersistentClient or HttpClient
     - collection: a string of the collection name
     - metadata: distance method of the embeddings,
                "hnsw:space": "l2" (squared L2)
                              "ip" (inner product")
                              "cosine" (cosine similarity)
     - embedding_function: default is to None, which calls
                        OpenAIEmbeddings default model text-ada02
     - overwrite: boolean, deletes the collection before recreating
       an empty version

    OUTPUT
    ------

    chromadb.Collection on which you can perform collection actions like
        get(), peek() and count()

    EXAMPLES
    --------

    # Basic
    client = galdre.connect_to_chroma()
    collection_name = "sample-index"
    collection = galdre.connect_to_collection(client, collection_name)

    # Overwriting
    collection = galdre.connect_to_collection(
        client,
        collection_name,
        overwrite=True
        )

    # Changing the distance method for embeddings
    collection = galdre.connect_to_collection(
        client,
        collection_name,
        metadata={"hnsw:space": "l2"}
        )

    # Changing the EmbeddingFunction
    sentence_transformer_ef = (
       embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2")
        )
    collection = connect_to_collection(
        client,
        collection_name,
        embedding_function=sentence_transformer_ef
        )

    # Operations on a Collection
    collection.get()
    collection.peek()
    collection.count()


    """
    if overwrite:
        client.delete_collection(name=collection)

    if embedding_function is None:
        embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-ada-002", api_key=openai_api_key
        )
    return client.get_or_create_collection(
        name=collection,
        metadata=metadata,
        embedding_function=embedding_function)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def add_to_collection(collection: chromadb.Collection,
                      data: Dict[str, list]) -> None:
    """
    DESCRIPTION
    -----------

    adds data to a collection, but adds a retry/backoff component
      to remain under the token limits

    INPUT
    -----

    collection: an already instantiated Collection type
    data: dict setup as
            {
                "documents": [str,..],
                "metadatas": [dict,..],
                "ids": [str|int,..]
            }

    OUTPUT
    ------

    None

    """
    try:
        collection.add(**data)
    except Exception as e:
        print(e)


def generate_uuids(num_ids: int, id_length: int = 12) -> List[str]:
    """
    DESCRIPTION
    -----------

    generates uuids based on the desired ids and the length of each id

    INPUT
    -----

    num_ids: int, number of ids to be generated
    id_length: int, how long those ids should be

    OUTPUT
    ------

    a list, where num_id=2,
    ['1e8f1d67-ef1', '192d0f25-b57']

    """
    return [str(uuid.uuid4())[:id_length] for _ in range(num_ids)]


def format_data(
    data: List[Document], ids: list = None, max_token: int = 8191
) -> Tuple[Dict[str, list], list]:
    """
    DESCRIPTION
    -----------

    converts list of Documents into the format needed to be uploaded
    into the collection

    INPUT
    -----

    data: list of langchain Document types
    ids: defaults to None, auto generates ids if none provided
    max_token: limit is set to 8191, which is the limit for
    OpenAI model "text-embedding-ada-002"

    OUTPUT
    ------

    Tuple of the dictionary of data to be uploaded and a list of Documents
    that were over the max_token limit

    data = [Document(page_content="blah, blah blah",
    metadata={"source": "source1.txt"}), ..]
    returns Tuple(
        {
            "documents": ["blah, blah blah", ..],
            "ids": ['1e8f1d67-ef1',],
            "metadatas":[{"source": "source1.txt"},..]
        },
        [
            ["over the limit page content", ..],
            [{"source2": "meta_over_limit.txt"}, ..]
        ]
    )

    """
    # get documents over token limit
    documents = [d.page_content for d in data]
    tokens = get_token_num(documents)
    indices_below, indices_above = extract_limit_indices(
        tokens, max_token=max_token)
    documents_below, documents_above = parse_data_by_limit(
        documents, indices_below, indices_above
    )

    # separate metadatas
    metadatas = [d.metadata for d in data]
    metadatas_below, metadatas_above = parse_data_by_limit(
        metadatas, indices_below, indices_above
    )

    ids = generate_uuids(len(documents_below)) if ids is None else ids
    data_to_add = {
        "documents": list(documents_below),
        "metadatas": list(metadatas_below),
        "ids": ids,
    }
    over_limit = [list(documents_above), list(metadatas_above)]
    return data_to_add, over_limit


def num_tokens_from_string(
        string: str,
        encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_token_num(data: List[str]):
    """Returns a list of the number of tokens from a list of strings"""
    return np.array([num_tokens_from_string(x) for x in data])


def extract_limit_indices(tokens: np.array, max_token: int = 8191) -> tuple:
    """Returns two lists for indices of tokens below or equal
      to the max_token limit and another list for those above"""
    indices_below = np.asarray(tokens <= max_token).nonzero()[0].flatten()
    indices_above = np.asarray(tokens > max_token).nonzero()[0].flatten()
    return list(indices_below), list(indices_above)


def parse_data_by_limit(data, indices_below, indices_above):
    """Returns two lists of strings, one for strings with indices below,
    one for strings with indices above the max token limit"""
    data = np.array(
        data, dtype=object) if not isinstance(
        data, np.array) else data
    return data[indices_below].flatten(), data[indices_above].flatten()


def upload_to_collection(
        collection: chromadb.Collection,
        data: List[Document],
        split_size: int = 100) -> tuple:
    """
    uploads data to a collection
    returns a list documents that were not uploaded
      since over the max token limit"""
    over_limit = []
    total_size = len(data)
    for i in range(0, total_size, split_size):
        data_chunk = data[i: i + split_size]
        formatted_chunk, over_limit_chunk = format_data(data_chunk)
        if len(over_limit_chunk) > 0:
            over_limit.append(over_limit_chunk)
        try:
            add_to_collection(collection, formatted_chunk)
        except Exception as e:
            print(e)
    return over_limit
