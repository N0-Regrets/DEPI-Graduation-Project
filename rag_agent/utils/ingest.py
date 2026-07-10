from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict


LAWS = [
    ("constitution", "./documents/دستور-جمهورية-مصر-العربية-2019.pdf"),
    ("commercial_law", "./documents/قانون التجارة 17 لسنة 1999 وتعديلاته.pdf"),
    ("penal_code", "./documents/قانون العقوبات طبقا ألحدث التعديالت بالقانون 54 لسنة 2003م.pdf"),
    ("civil_procedures", "./documents/قانون رقم ۱۳ لسنة ۱۹٦۸ بإصدار قانون المرافعات المدنية والتجارية وفقاً لآخر تعديل صادر في 10 يولية عام 2024..pdf"),
]


docs_list = []
for law_name, path in LAWS:
    for page in PyPDFLoader(path).load():
        page.metadata["law_name"] = law_name
        docs_list.append(page)

print(f"Total pages loaded: {len(docs_list)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=128,
)

doc_splits = text_splitter.split_documents(docs_list)

print(f"Total chunks: {len(doc_splits)}")


embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

Chroma.from_documents(
    documents = doc_splits,
    embedding = embedding_model,
    persist_directory = "./chroma_db",
)

print("Vector store saved to ./chroma_db")



