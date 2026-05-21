import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from rag_agent.utils.state import GraphState

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
llm = ChatOpenRouter(                                     
    model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    api_key = os.getenv("OPENROUTER_API_KEY2"),
    temperature = 0
)
prompt_ar = ChatPromptTemplate.from_template("""
أنت مساعد قانوني متخصص في القانون المصري، لديك معرفة عميقة بالدستور المصري، والقانون المدني، وقانون العقوبات، وقانون التجارة، وقانون الإجراءات الجنائية والمدنية.

قواعد يجب الالتزام بها:
- أجب بناءً على المقتطفات القانونية المقدمة فقط، ولا تعتمد على معلومات خارجها.
- اذكر رقم المادة القانونية عند الإمكان (مثال: وفقاً للمادة ٥٤ من الدستور...).
- إذا كان السؤال يتعلق بأكثر من قانون، اذكر المصدر لكل جزء من إجابتك.
- إذا كانت المقتطفات لا تكفي للإجابة بشكل كامل، صرّح بذلك واذكر ما يمكن الإجابة عنه فقط.
- لا تخمّن ولا تستنتج أحكاماً غير موجودة في النص.
- اكتب إجابتك بأسلوب واضح ومنظم.

المقتطفات القانونية:
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
You are a classifier for a legal assistant specialized in Egyptian law.
Determine if the question is related to Egyptian law, the Egyptian constitution, legal rights, court procedures, contracts, crimes, or any legal matter in Egypt.
Greetings, general knowledge questions, and non-legal topics are off_topic.
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
        return {**state, "answer": state.get("answer") or "لا تتوفر معلومات كافية في المستندات للإجابة على هذا السؤال."}
    context = "\n\n".join(d.page_content for d in state["documents"])
    chain = prompt_ar | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "answer": answer.content}