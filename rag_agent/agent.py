from langgraph.graph import StateGraph, END
from rag_agent.utils.state import GraphState
from rag_agent.utils.nodes import retrieve_node, generate_node
from langchain_chroma import Chroma

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

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
    
    print("Answer:", result["answer"])
    
