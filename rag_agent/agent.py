from langgraph.graph import StateGraph, END
from rag_agent.utils.state import GraphState
from rag_agent.utils.nodes import intent_node, reject_node, retrieve_node, generate_node
from langchain_chroma import Chroma

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
    "ما هي الحقوق الأساسية التي يكفلها الدستور المصري للمواطنين؟",
    "ما هي صلاحيات رئيس الجمهورية في الدستور المصري؟",
    "كيف يضمن الدستور استقلالية القضاء في مصر؟",
    "ما هي اختصاصات المحكمة الدستورية العليا؟",
    "ما هي شروط تعديل الدستور المصري؟",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)
    
    result = app.invoke({"question": question})
    
    print(f"\n📄 Retrieved {len(result['documents'])} document(s):")
    for i, doc in enumerate(result["documents"], 1):
        print(f"\n  [{i}] Source: {doc.metadata.get('source', 'unknown')}")
        print(f"       Content: {doc.page_content}...")
    
    print("\nAnswer:", result["answer"])
    
