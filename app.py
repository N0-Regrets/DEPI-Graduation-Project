import streamlit as st
from rag_agent.agent import build_graph

st.set_page_config(
    page_title="المساعد القانوني المصري",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ المساعد القانوني المصري")
st.caption("اسأل عن الدستور المصري، القانون المدني، قانون العقوبات، قانون التجارة، وقوانين الإجراءات")

@st.cache_resource
def load_graph():
    return build_graph()

app = load_graph()

LAW_NAME_AR = {
    "constitution":        "الدستور المصري",
    "civil_code":          "القانون المدني",
    "penal_code":          "قانون العقوبات",
    "commercial_law":      "قانون التجارة",
    "criminal_procedures": "قانون الإجراءات الجنائية",
    "civil_procedures":    "قانون المرافعات المدنية",
}

SAMPLE_QUESTIONS = [
    "ما هي الحقوق الأساسية التي يكفلها الدستور المصري للمواطنين؟",
    "ما هي صلاحيات رئيس الجمهورية في الدستور المصري؟",
    "ما هي عقوبة السرقة في قانون العقوبات المصري؟",
    "ما هي شروط صحة العقد في القانون المدني المصري؟",
    "ما هي حقوق المتهم في قانون الإجراءات الجنائية؟",
    "ما هي اختصاصات المحكمة الدستورية العليا؟",
]

st.subheader("أسئلة مقترحة")
cols = st.columns(3)
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 3].button(q, key=f"sample_{i}", use_container_width=True):
        st.session_state["pending_question"] = q

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

col_chat, col_clear = st.columns([8, 1])
with col_clear:
    if st.button("🗑️ مسح", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 المصادر المسترجعة"):
                for j, doc in enumerate(msg["sources"], 1):
                    law_name = doc.metadata.get("law_name", "")
                    law_ar = LAW_NAME_AR.get(law_name, law_name)
                    page = doc.metadata.get("page_label") or doc.metadata.get("page", "")
                    st.markdown(f"**المقطع {j}** — {law_ar} — صفحة {page}")
                    st.info(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("اكتب سؤالك هنا...") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والإجابة..."):
            result = app.invoke({
                "question": user_input,
                "documents": [],
                "answer": "",
                "intent": "",
            })

        answer = result.get("answer") or "حدث خطأ أثناء معالجة سؤالك، يرجى المحاولة مرة أخرى."
        docs = result.get("documents", [])

        st.markdown(answer)

        if docs:
            with st.expander("📄 المصادر المسترجعة"):
                for j, doc in enumerate(docs, 1):
                    law_name = doc.metadata.get("law_name", "")
                    law_ar = LAW_NAME_AR.get(law_name, law_name)
                    page = doc.metadata.get("page_label") or doc.metadata.get("page", "")
                    st.markdown(f"**المقطع {j}** — {law_ar} — صفحة {page}")
                    st.info(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
    })
