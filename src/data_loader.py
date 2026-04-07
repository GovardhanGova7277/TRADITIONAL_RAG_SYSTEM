from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader, 
    Docx2txtLoader, 
    JSONLoader,
    UnstructuredPowerPointLoader # Added for .pptx
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader


from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    Docx2txtLoader, UnstructuredPowerPointLoader
)

def load_all_documents(data_dir: str) -> List[Any]:
    """
    A generic loader that automatically detects file types and 
    uses the appropriate LangChain loader.
    """
    data_path = Path(data_dir).resolve()
    documents = []
    
    # Map extensions to their specific loader classes
    LOADER_MAPPING = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader
    }

    # Iterate through all files in the directory
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            print(file_path)
            ext = file_path.suffix.lower()
            print(ext)
        
            if ext in LOADER_MAPPING:
                print(f"[DEBUG] Loading {ext.upper()}: {file_path.name}")
                try:
                    # Initialize the specific loader for this file type
                    loader_class = LOADER_MAPPING[ext]
                    loader = loader_class(str(file_path))
                    
                    # Load the data and add to our master list
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                    print(f"[DEBUG] Success: Loaded {len(loaded_docs)} chunks")
                    
                except Exception as e:
                    print(f"[ERROR] Failed to load {file_path.name}: {e}")
            else:
                print(f"[SKIP] Unsupported format: {file_path.name}")
        else:
            print(f"[INFO] Skipping directory: {file_path.name}")

    print(f"\n[TOTAL] Consolidated {len(documents)} document chunks into pipeline.")
    return documents