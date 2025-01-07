import os
from pathlib import Path

UPDATE = False

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
    當不知道如何回答時可以使用這個程式，由環保局統整常見QA回答民眾問題，當民眾有詢問相關內容時請將關鍵字放入其中，該函示會回傳相關文件內容。

    Args:
        question: A user question (FAQ style).

    Returns:
        str: A short text snippet that is most relevant to the user question.
    """

    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[1] 
    DOCUMENTS_PATH = ROOT / "documents"
    
    file_paths = [
        str(DOCUMENTS_PATH  / f)  # 或 (DOCUMENTS_PATH / f) 取決於你的 Path 寫法
        for f in os.listdir(DOCUMENTS_PATH)
        if f.lower().endswith(".pdf")  # 只篩選 .pdf 檔案
    ]
    
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
    vector_store = Chroma(
        embedding_function= embeddings,
        persist_directory="./chroma_langchain_db"
    )
    
    if UPDATE:
        file_paths = [
            str(DOCUMENTS_PATH  / f)  # 或 (DOCUMENTS_PATH / f) 取決於你的 Path 寫法
            for f in os.listdir(DOCUMENTS_PATH)
            if f.lower().endswith(".pdf")  # 只篩選 .pdf 檔案
        ]
        
        loader = UnstructuredLoader(
            file_path= file_paths,  
            chunking_strategy="basic",
            max_characters=10000,
            include_orig_elements=False,
        )
    # Extend our full doc list with the docs from each file
        all_docs = loader.load()
        all_docs  = filter_complex_metadata(all_docs)
        vector_store.add_documents(all_docs)
    
    # 5. Query the vector store
    #    Optionally, you might want to limit the number of results, e.g. k=3
    results = vector_store.similarity_search(question, k=1)

    if not results:
        return "No relevant information found."
    
    # Return the content of the best chunk
    return [results[0].page_content]

if __name__ == "__main__":
    # Simple test
    question = "您好我想詢問一下室內空氣品質維護管理計劃書有沒有格式可以下載"
    answer = get_rag_faq(question)
    print("FAQ Answer:\n",len(answer) , answer)
