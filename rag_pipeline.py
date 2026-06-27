import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def setup_rag_pipeline():
    docs_dir = os.path.join(os.path.dirname(__file__), "documents")
    
    # Load all text files from the documents directory
    loader = DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Using ephemeral Chroma DB in memory for this project (or persistent if needed)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    return retriever

# Global retriever instance
retriever = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = setup_rag_pipeline()
    return retriever

def retrieve_context(query: str) -> str:
    ret = get_retriever()
    docs = ret.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)
