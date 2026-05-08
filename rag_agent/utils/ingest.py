from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.document_loaders import UnstructuredMarkdownLoader

pdf_paths = [
    "./documents/دستور-جمهورية-مصر-العربية-2019.pdf",
    
    
]

docs = [PyPDFLoader(path).load() for path in pdf_paths]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=250,
)
doc_splits = text_splitter.split_documents(docs_list)


print(f"Total chunks: {len(doc_splits)}")

# Preview first few chunks
for i, chunk in enumerate(doc_splits[:50]):
    print(f"\n--- Chunk {i} ---")
    print(f"Length: {len(chunk.page_content)} chars")
    print(f"Metadata: {chunk.metadata}")
    print(chunk.page_content)
    
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)



Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    persist_directory="./chroma_db"  
)


