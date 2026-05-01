from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from rag_agent.utils.state import GraphState

# --- Retriever setup ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever()

    # Grab your retriever from wherever you defined it


# --- LLM + Prompt setup ---
llm = ChatOpenRouter(                                     
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key=("USE_YOUR_API_KEY"),
    temperature=0
)
prompt_ar = ChatPromptTemplate.from_template("""
أنت مساعد قانوني متخصص في القانون المصري. أجب على السؤال التالي بناءً على المقتطفات القانونية المقدمة فقط.

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
def retrieve_node(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}

def generate_node(state: GraphState) -> GraphState:
    context = "\n\n".join(d.page_content for d in state["documents"])
    chain = prompt_ar | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "answer": answer.content}