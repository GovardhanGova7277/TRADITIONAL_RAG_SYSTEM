from data_loader import load_all_documents
from embeddings import EmbeddingPipeline
from vector_store import FaissVectorStore
from search import RAGSearch

import streamlit as st
import os
from pathlib import Path

# Page Config
st.set_page_config(page_title="RAG Intelligence System", layout="wide")
st.title("📄 Multi-Format RAG Document Assistant")

# Initialize RAG Pipeline [cite: 348, 479]
@st.cache_resource
def init_rag():
    return RAGSearch()

rag_engine = init_rag()

# Sidebar: File Upload
with st.sidebar:
    st.header("Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, DOCX, or PPTX", 
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx', 'pptx']
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            # 1. Define base data directory
            base_data_dir = Path("live_data")
            
            # 2. Map extensions to specific subdirectories
            extension_map = {
                ".pdf": "pdf",
                ".txt": "text_files",
                ".docx": "docx",
                ".pptx": "pptx",
                ".csv": "csv"
            }

            for uploaded_file in uploaded_files:
                # Get the extension (e.g., '.pdf')
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                # Determine sub-folder based on extension map, default to 'others'
                sub_folder = extension_map.get(file_ext, "others")
                target_dir = base_data_dir / sub_folder
                
                # Create the specific sub-directory if it doesn't exist
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the file to data/sub_folder/filename
                save_path = target_dir / uploaded_file.name
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            with st.spinner("Chunking and Indexing..."):
                # Pass the base "data" folder; load_all_documents handles the subfolders
                docs = load_all_documents("live_data")
                rag_engine.vectorstore.build_from_documents(docs)
                st.success(f"Successfully organized and indexed {len(docs)} documents!")
        else:
            st.warning("Please upload files first.")

# Main Chat Interface [cite: 402]
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching and Summarizing..."):
            # Execute RAG query flow [cite: 335, 480]
            response = rag_engine.search_and_summarize(query)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
# if __name__ == "__main__":

#     ## The faiss.index File (The Mathematical Vector Space)
#     ## The metadata.pkl File (The Python Object Storage) - Dim 284
#     # list_of_docs = load_all_documents("data")
#     # chunks = EmbeddingPipeline(chunk_size=200,chunk_overlap=20).chunk_documents(list_of_docs)
#     # vectors  = EmbeddingPipeline().embed_chunks(chunks)
#     # store = FaissVectorStore("faiss_store")
#     # store.load()
#     # print(store.query("List out all the kubernetes architecture components with clear explaination ?", top_k=3))
#     # store.build_from_documents(list_of_docs)
#     rag = RAGSearch()
#     summary  = rag.search_and_summarize("Explain about all the kubernetes cluster components ?",top_k=3)
#     print("Summary : ",summary)
    
    