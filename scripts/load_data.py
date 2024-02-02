from toddbo import connect_to_chroma, load_documents_to_chroma

client = connect_to_chroma()
load_documents_to_chroma(client)
