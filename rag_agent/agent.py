from langgraph.graph import StateGraph, END
from rag_agent.utils.state import GraphState
from rag_agent.utils.nodes import intent_node, safety_reject_node, out_of_scope_reject_node, clarification_node, retrieve_node, generate_node



def route_intent(state: GraphState) -> str:
    return state["intent"]

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("intent_check", intent_node)
    graph.add_node("safety_reject", safety_reject_node)
    graph.add_node("out_of_scope_reject", out_of_scope_reject_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)


    graph.set_entry_point("intent_check")
    graph.add_conditional_edges("intent_check", route_intent,
    

     {
        "legal_question": "retrieve",
        "clarification_needed": "clarification",
        "out_of_scope": "out_of_scope_reject",
        "unsafe_or_disallowed": "safety_reject",
        "greeting_or_meta": "generate",
    })

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("clarification", END)
    graph.add_edge("out_of_scope_reject", END)
    graph.add_edge("safety_reject", END)

    return graph.compile()





    
