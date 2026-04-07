from data_loader import load_all_documents
from embeddings import EmbeddingPipeline
from vector_store import FaissVectorStore
from search import RAGSearch

if __name__ == "__main__":

    ## The faiss.index File (The Mathematical Vector Space)
    ## The metadata.pkl File (The Python Object Storage) - Dim 284
    # list_of_docs = load_all_documents("data")
    # chunks = EmbeddingPipeline(chunk_size=200,chunk_overlap=20).chunk_documents(list_of_docs)
    # vectors  = EmbeddingPipeline().embed_chunks(chunks)
    # store = FaissVectorStore("faiss_store")
    # store.load()
    # print(store.query("List out all the kubernetes architecture components with clear explaination ?", top_k=3))
    # store.build_from_documents(list_of_docs)
    rag = RAGSearch()
    summary  = rag.search_and_summarize("Explain about all the kubernetes cluster components ?",top_k=3)
    print("Summary : ",summary)
    
    