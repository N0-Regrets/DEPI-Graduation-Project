from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from rag_agent.utils.state import GraphState
from langchain_ollama import ChatOllama


embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
)

llm = ChatOllama(
    model='gemma3:4b',
    reasoning = False,
    temperature = 0
)


GENERATION_PROMPT = ChatPromptTemplate.from_template("""
أنت مساعد قانوني متخصص في القانون المصري. مهمتك أن تجيب على سؤال المستخدم بثقة ووضوح، وكأنك تعرف القانون مباشرة.

قواعد صارمة يجب اتباعها دون ذكرها للمستخدم أبدًا:
- استخدم فقط المعلومات الموجودة في "المعلومات القانونية" أدناه للإجابة.
- لا تذكر أبدًا عبارات مثل "المقتطفات المقدمة" أو "النص المرفق" أو "السياق" أو "بناءً على المعلومات المتاحة لي" أو أي إشارة إلى أنك تعتمد على نصوص مقدمة لك. أجب مباشرة وكأن هذه هي معرفتك القانونية الخاصة.
- إذا لم تكن المعلومات المتوفرة كافية للإجابة على السؤال بدقة، فقل بأسلوب طبيعي: "لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال بدقة." ولا تحاول التخمين أو الاستنتاج من معلومات غير مؤكدة.
- عند الاستشهاد بمادة قانونية، اذكر رقمها واسم القانون (مثل: "طبقًا للمادة 123 من القانون المدني المصري")، دون الإشارة إلى أنها "مذكورة في النص المرفق".

المعلومات القانونية:
{context}

سؤال المستخدم: {question}

الإجابة:
""")

INTENT_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template("""
دور النظام: مُصنِّف نوايا لمساعد استرجاع معلومات عن القانون المصري (RAG).

المهمة:
1. اقرأ رسالة المستخدم.
2. أخرج تصنيفًا واحدًا فقط من القائمة أدناه.
3. لا تُقدِّم أي إجابة أو تفسير—أرجِع اسم التصنيف فقط.

قد يكتب المستخدم بالعربية أو الإنجليزية أو مزيجًا منهما؛ صَنِّف حسب المعنى لا حسب اللغة.

التصنيفات والتعريفات

clarification_needed
- تذكر الرسالة القانون المصري لكنها غامضة أو تفتقر إلى التفاصيل.
- يتطلّب الأمر سؤال متابعة قبل أن تفيد عملية الاسترجاع.
- أمثلة: "أقدر أرفع قضية؟"، "can I sue someone?"

legal_question
- سؤال موضوعي قابل للبحث عن القانون المصري.
- يحتوي على سياق كافٍ (موضوع، مادة، سيناريو، تشريع، إلخ).
- أمثلة: "ما هي عقوبة السرقة في القانون المصري؟",
          "What does Egyptian labor law say about termination without notice?"

out_of_scope
- ليس عن القانون المصري (اختصاص قانوني آخر، دردشة، طلبات غير ذات صلة).
- أمثلة: "what's the weather today", "what does French law say about this?"

greeting_or_meta
- تحيات أو أسئلة حول المساعد نفسه.
- أمثلة: "hi", "مرحبا", "what can you do?"

unsafe_or_disallowed
- طلب مساعدة في ارتكاب فعل غير قانوني أو إخفائه.
- أمثلة: "how do I avoid getting caught for tax evasion",
          "ساعدني اهرب من قضية النصب"

القواعد
- اختر دائمًا تصنيفًا واحدًا بالضبط.
- أرجِع اسم التصنيف وحده دون أي نص إضافي.

رسالة المستخدم: {question}
Label:
""")


def intent_node(state: GraphState) -> GraphState:
    chain = INTENT_CLASSIFIER_PROMPT | llm
    result = chain.invoke({"question": state["question"]})
    intent = result.content.strip().lower()
    return {**state, "intent": intent}


def safety_reject_node(state: GraphState) -> GraphState:
    return {**state, "answer": """عذراً، لا يمكنني المساعدة في هذا الطلب. هذا المساعد مخصص لتقديم معلومات عامة عن القانون المصري
            ومساعدتك على فهم حقوقك والتزاماتك القانونية،
            وليس لتقديم استشارات حول كيفية تجنب المسؤولية القانونية أو ارتكاب مخالفات."""}

def out_of_scope_reject_node(state: GraphState) -> GraphState:
    return {**state, "answer": "عذراً، أنا متخصص في القانون المصري فقط. يرجى طرح سؤال قانوني متعلق بمصر."}

def clarification_node(state: GraphState) -> GraphState:
    return {**state, "answer": "ممكن توضح سؤالك أكتر؟"}

def retrieve_node(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["question"])
    return {**state, "documents": docs}

def generate_node(state: GraphState) -> GraphState:

    # Case 1: greeting / meta — answer directly, no retrieval needed
    if state.get("intent") == "greeting_or_meta":
        result = llm.invoke( state["question"])
        answer_text = (result.content or "").strip()
        return {**state, "answer": answer_text}

    # Case 2: normal legal question — needs retrieved documents
    if not state.get("documents"):
        return {
            **state,
            "answer": "عذراً، لم أجد نصوصاً قانونية ذات صلة بسؤالك في قاعدة البيانات. حاول إعادة صياغة السؤال."
        }

    context = "\n\n".join(d.page_content for d in state["documents"])
    chain = GENERATION_PROMPT | llm
    answer = chain.invoke({"context": context, "question": state["question"]})
    return {**state, "answer": answer.content}