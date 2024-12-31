import os
from pathlib import Path

# LangChain / local modules
from langchain_unstructured import UnstructuredLoader  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_core.tools import tool  # type: ignore
from langchain_community.vectorstores.utils import filter_complex_metadata  # type: ignore

# LangChain text splitter (optional, but recommended for better chunking)
from langchain.text_splitter import RecursiveCharacterTextSplitter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
DOCUMENTS_PATH = ROOT / "documents"

@tool(parse_docstring=True)
def get_rag_faq(question: str):
    """
    Answers FAQ-style questions by:
    1. Loading documents from the `documents` folder (one by one).
    2. Chunking the documents.
    3. Creating a vector store with HuggingFace embeddings (Chroma).
    4. Querying the vector store for the most relevant chunk.
    
    Args:
        question: A user question (FAQ style).

    Returns:
        str: A short text snippet that is most relevant to the user question.
    """

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1] 
    DOCUMENTS_PATH = ROOT / "documents"
    
    # 1. Gather all file paths from your /documents directory
    file_paths = [str(DOCUMENTS_PATH / f) for f in os.listdir(DOCUMENTS_PATH)]
    
    loader = UnstructuredLoader(
        file_path= file_paths,  
        chunking_strategy="basic",
        max_characters=1000,
        include_orig_elements=False,
    )
    # Extend our full doc list with the docs from each file
    all_docs = loader.load()
    all_docs  = filter_complex_metadata(all_docs)
    
    # 3. Initialize embeddings
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cpu"}  # change to "cuda" if you have GPU
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    # 4. Create a Chroma vector store and add docs
    vector_store = Chroma(embedding_function=embeddings)
    vector_store.add_documents(all_docs)
    
    # 5. Query the vector store
    #    Optionally, you might want to limit the number of results, e.g. k=3
    results = vector_store.similarity_search(question, k=1)

    if not results:
        return "No relevant information found."
    
    # Return the content of the best chunk
    return results[0].page_content

if __name__ == "__main__":
    # Simple test
    question = "電動機車補助政策?"
    answer = get_rag_faq(question)
    print("FAQ Answer:\n", answer)
