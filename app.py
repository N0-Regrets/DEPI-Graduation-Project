# app_streamlit.py
import streamlit as st
from rag_agent.agent import build_graph

st.set_page_config(
    page_title="Arabic Legal RAG Agent",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ المساعد القانوني المصري")
st.caption("اسأل عن الدستور المصري وسيجيبك النظام بناءً على المستندات القانونية")

# ── Cache the compiled graph so it's built only once ──────────────────────────
@st.cache_resource
def load_graph():
    return build_graph()

app = load_graph()

# ── Suggested questions ────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "ما هي المصادر القانونية التي يجب على القاضي اللجوء إليها في حال عدم وجود نص تشريعي يمكن تطبيقه على الواقعة؟",  # القانون المدني
    "ما هي الجهة التي يختص القانون بمنحها دون غيرها سلطة رفع الدعوى الجنائية ومباشرتها؟",                          # قانون الإجراءات الجنائية
    "ما هي الدفاتر التجارية الإلزامية التي يجب على كل تاجر تجاوز رأس ماله المستثمر عشرين ألف جنيه أن يمسكها؟",    # القانون التجاري
    "كيف يتم التمييز بين الجنايات والجنح والمخالفات بناءً على نوع العقوبات المقررة لكل منها؟",                      # قانون العقوبات
    "ما هي الشروط التي يجب توافرها في المصلحة لكي تُقبل أي دعوى أو طلب أو دفع أمام المحاكم؟",                     # قانون المرافعات
    "ما هي الحقوق الأساسية التي يكفلها الدستور المصري للمواطنين؟",                                                  # الدستور
]

st.subheader("أسئلة مقترحة")
cols = st.columns(2)
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
        st.session_state["pending_question"] = q

# ── Chat history ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 المصادر المسترجعة"):
                for j, doc in enumerate(msg["sources"], 1):
                    st.markdown(f"**المقطع {j}:**")
                    st.info(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

# ── Input: typed or clicked sample ────────────────────────────────────────────
pending = st.session_state.pop("pending_question", None)
user_input = st.chat_input("اكتب سؤالك هنا...") or pending

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("جاري البحث والإجابة..."):
            result = app.invoke({"question": user_input})
        
        answer = result["answer"]
        docs   = result.get("documents", [])   # expose retrieved docs if your state includes them

        st.markdown(answer)

        if docs:
            with st.expander("📄 المصادر المسترجعة"):
                for j, doc in enumerate(docs, 1):
                    st.markdown(f"**المقطع {j}:**")
                    st.info(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": docs,
    })