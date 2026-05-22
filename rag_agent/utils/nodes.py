import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from rag_agent.utils.state import GraphState
from langchain_openai import AzureChatOpenAI

load_dotenv()

# --- Retriever setup ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"}
)

retriever = vectorstore.as_retriever()


# --- LLM + Prompt setup ---
llm = AzureChatOpenAI(                                     
    azure_deployment = "gpt-4.1-mini",
    azure_endpoint = "https://azure-openai-api1256.openai.azure.com/openai/deployments/gpt-4.1-mini/chat/completions?api-version=2025-01-01-preview",
    api_key = "5JqKdaFCgrp4Kx1pSWKAtUWyXLvlaSITrUXgImio62ZGmzyRvWSAJQQJ99CEACF24PCXJ3w3AAABACOG9hR4",
    api_version = "2024-12-01-preview",
    temperature=0
)

prompt_ar = ChatPromptTemplate.from_template("""
أنت مساعد قانوني متخصص في القانون المصري. أجب على السؤال التالي بناءً على المقتطفات القانونية المقدمة فقط.
إذا كانت المقاطع المتوفرة لا تكفي للإجابة، قل صراحةً: لا تتوفر معلومات كافية ولا تخمّن.

السياق:
{context}

السؤال: {question}

الإجابة:
""")

prompt_en = ChatPromptTemplate.from_template("""
You are a legal assistant specialized in Egyptian law.
Answer the following question based ONLY on the provided legal excerpts.
Answer in Arabic. Be precise and cite the relevant legal articles when possible.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer (in Arabic):
""")

# --- Nodes ---
_INTENT_PROMPT = ChatPromptTemplate.from_template("""
You are a classifier. Determine if the following question is related to Egyptian law, the Egyptian constitution, or Egyptian legal matters.
Respond with exactly one word: on_topic or off_topic

Question: {question}
""")

def intent_node(state: GraphState) -> GraphState:
    chain = _INTENT_PROMPT | llm
    result = chain.invoke({"question": state["question"]})
    intent = result.content.strip().lower()
    if intent not in ("on_topic", "off_topic"):
        intent = "on_topic"
    return {**state, "intent": intent}

def reject_node(state: GraphState) -> GraphState:
    return {**state, "answer": "عذراً، أنا متخصص في القانون المصري فقط. يرجى طرح سؤال قانوني متعلق بمصر."}

def retrieve_node(state: GraphState) -> GraphState:
    docs = vectorstore.similarity_search(state["question"], k=5)
    return {**state, "documents": docs}

def generate_node(state: GraphState) -> GraphState:
    if not state.get("documents"):
        return state
    context = "\n\n".join(d.page_content for d in state["documents"])
    chain = prompt_ar | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "answer": answer.content}