from langchain.vectorstores import Chroma


def generate_context(
    client,
    embedding_function,
    collection_name="resume",
    search_type="mmr",
    search_kwargs={"fetch_k": 100},
):
    docsearch = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    retriever = docsearch.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever
