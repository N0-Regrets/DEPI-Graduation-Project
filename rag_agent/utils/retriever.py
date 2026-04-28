from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool


pdf_paths = [
    "./documents/دستور-جمهورية-مصر-العربية-2019.pdf",

]

docs = [PyPDFLoader(path).load() for path in pdf_paths]




docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)



embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    persist_directory="./chroma_db"  
)

retriever = vectorstore.as_retriever()


@tool
def retrieve_from_law_files(query: str) -> str:
    """Search and retrieve relevant articles and provisions from Egyptian legal documents.
    
    The knowledge base includes:
    - Egyptian Constitution
    - Civil Code (Law 131/1948) — contracts, property, liability
    - Penal Code (Law 58/1937) — crimes and punishments
    - Commercial Code (Law 17/1999) — business and trade
    - Civil & Commercial Procedure Code — court operations and filing
    - Criminal Procedure Code (Law 150/1950) — arrests, trials, evidence
    - Personal Status Law — marriage, divorce, and inheritance
    
    Use this tool when the user asks any question related to Egyptian law, 
    legal rights, penalties, court procedures, or personal status matters.
    Documents are in Arabic.
    """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
retriever_tool = retrieve_from_law_files

docs = retriever.invoke("ما هي الحقوق الأساسية التي يكفلها الدستور المصري للمواطنين؟")
for i, doc in enumerate(docs):
    print(f"--- DOC {i} ---")
    print(repr(doc.page_content))
    print(doc.metadata)