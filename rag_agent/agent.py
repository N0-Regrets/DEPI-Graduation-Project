from langgraph.graph import StateGraph, END
from rag_agent.utils.state import GraphState
from rag_agent.utils.nodes import retrieve_node, generate_node,intent_node, reject_node
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy  
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_openai import ChatOpenAI


def _route_intent(state: GraphState) -> str:
    return state["intent"]

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("intent_check", intent_node)
    graph.add_node("reject", reject_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("intent_check")
    graph.add_conditional_edges("intent_check", _route_intent, {
        "on_topic": "retrieve",
        "off_topic": "reject",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("reject", END)

    return graph.compile()

app = build_graph()


questions = [
    #الدستور المصري 
    "ما هي الحقوق الأساسية التي يكفلها الدستور المصري للمواطنين؟",
    "ما هي صلاحيات رئيس الجمهورية في الدستور المصري؟",
    "كيف يضمن الدستور استقلالية القضاء في مصر؟",
    
    # القانون المدني
    "ما هي المصادر القانونية التي يجب على القاضي اللجوء إليها في حال عدم وجود نص تشريعي يمكن تطبيقه على الواقعة؟",
    "متى يُعتبر استعمال الشخص لحقه غير مشروع وفقاً لمواد القانون المدني؟",
    "ما هو السن القانوني الذي حدده القانون لبلوغ سن الرشد واكتساب الأهلية الكاملة لمباشرة الحقوق المدنية؟",
    
    # قانون الإجراءات الجنائية
    "ما هي الجهة التي يختص القانون بمنحها دون غيرها سلطة رفع الدعوى الجنائية ومباشرتها؟",
    "ما هو الميعاد المحدد لمأمور الضبط القضائي لإرسال المتهم المقبوض عليه إلى النيابة العامة، وما هي المدة الممنوحة للنيابة لاستجوابه؟",
    "ما هي الحالات المحددة التي يجوز فيها طلب إعادة النظر في الأحكام النهائية الصادرة بالعقوبة في مواد الجنايات والجنح؟",
    
    # القانون التجاري
    "ما هي الدفاتر التجارية الإلزامية التي يجب على كل تاجر تجاوز رأس ماله المستثمر عشرين ألف جنيه أن يمسكها؟",
    "ما هي البيانات الستة الأساسية التي يجب أن يشتمل عليها الشيك ليعتبر صحيحاً من الناحية القانونية؟",
    "كم تبلغ المدة الزمنية لحظر مزاولة نشاط مماثل عند التصرف في المتجر بالبيع أو التأجير؟",
    
    # قانون العقوبات
    "كيف يتم التمييز بين الجنايات والجنح والمخالفات بناءً على نوع العقوبات المقررة لكل منها؟",
    "ما هو التعريف القانوني للشروع في الجريمة، وهل يُعاقب القانون على مجرد العزم أو الأعمال التحضيرية؟",
    "ما المقصود بالإصرار السابق الذي يوجب عقوبة الإعدام في جرائم القتل العمد؟",
    
    # قانون المرافعات المدنية والتجارية
    "ما هي الشروط التي يجب توافرها في المصلحة لكي تُقبل أي دعوى أو طلب أو دفع أمام المحاكم؟",
    "ما هو النصاب المالي الذي يحدد الاختصاص النوعي لمحكمة المواد الجزئية في الدعاوى المدنية والتجارية؟",
    "متى تختص محاكم الجمهورية بنظر الدعاوى المرفوعة ضد الأجانب الذين ليس لديهم موطن أو محل إقامة في مصر؟",
]

#  Collect outputs 
all_questions = []
all_answers   = []
all_contexts  = []

# for question in questions:
#     print(f"\n{'='*60}")
#     print(f"Question: {question}")
#     print('='*60)

#     result = app.invoke({"question": question})

#     # Collect for RAGAS
#     all_questions.append(question)
#     all_answers.append(result["answer"])
#     all_contexts.append([doc.page_content for doc in result["documents"]])  # list of strings

#     # Your existing print logic
#     print(f"\n📄 Retrieved {len(result['documents'])} document(s):")
#     for i, doc in enumerate(result["documents"], 1):
#         print(f"\n  [{i}] Source: {doc.metadata.get('source', 'unknown')}")
#         print(f"       Content: {doc.page_content}...")
#     print("\nAnswer:", result["answer"])
    
    
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# )

# llm = ChatOpenAI(
#     model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
#     api_key="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",
#     temperature=0,
# )

# ragas_llm        = LangchainLLMWrapper(llm)
# ragas_embeddings = LangchainEmbeddingsWrapper(embedding_model)

# # ── Instantiate metrics with LLM/embeddings ───────────────
# faithfulness_metric     = Faithfulness(llm=ragas_llm)
# answer_relevancy_metric = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

# # ── Build dataset ─────────────────────────────────────────
# dataset = Dataset.from_dict({
#     "question": all_questions,
#     "answer":   all_answers,
#     "contexts": all_contexts,
# })

# # ── Evaluate ──────────────────────────────────────────────
# result = evaluate(dataset, metrics=[faithfulness_metric, answer_relevancy_metric])

# print(result)


    
