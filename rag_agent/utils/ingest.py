from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

pdf_paths = [
    "./documents/دستور-جمهورية-مصر-العربية-2019.pdf",
]

docs = [PyPDFLoader(path).load() for path in pdf_paths]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)



Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    persist_directory="./chroma_db"  
)


