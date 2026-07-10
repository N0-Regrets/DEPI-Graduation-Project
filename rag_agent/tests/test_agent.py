import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.models import OpenRouterModel
from rag_agent.agent import build_graph
from deepeval.evaluate import AsyncConfig
import json



app = build_graph()

questions = [
    # دستور جمهورية مصر العربية
    "ما هي الشروط والضوابط التي يحددها الدستور لمن يرغب في الترشح لمنصب رئيس الجمهورية؟",
    "كيف حدد الدستور مكانة الأزهر الشريف والضمانات المتعلقة باستقلال شيخه؟",
    "ما هي نسبة تمثيل المرأة المقررة في مجلس النواب، وما هي شروط الترشح لعضوية المجلس؟",
    "ما هي الالتزامات التي يفرضها الدستور على الدولة تجاه حماية نهر النيل والحقوق التاريخية المتعلقة به؟",
    "ما هو الحد الأدنى لعدد أعضاء مجلس النواب؟",

    # قانون التجارة
    "ما هي المعايير التي يحددها القانون لاكتساب الشخص صفة التاجر؟",
    "ما هي الدفاتر التجارية التي يلتزم التاجر بإمساكها، وما هي الشروط الواجب توافرها في تنظيمها؟",
    "ما هي البيانات الأساسية والجوهرية التي يجب أن تشتمل عليها الكمبيالة ليعتد بها قانوناً؟",
    "متى يسقط حق المشتري في إقامة دعوى الفسخ أو إنقاص الثمن بسبب وجود عيب في البضاعة أو نقص في كميتها؟",
    "ما هي الآثار القانونية المترتبة على تظهير الشيك على بياض، وكيف يمكن للحامل التصرف فيه؟",

    # قانون العقوبات
    "كيف صنف القانون أنواع الجرائم وما هي العقوبات الأصلية المقررة لكل نوع منها؟",
    "ما هو التعريف القانوني لمفهومي سبق الإصرار والترصد في سياق جرائم القتل؟",
    "ما هي العقوبة المقررة للموظف العمومي الذي يطلب أو يقبل عطية مقابل الامتناع عن أداء عمل من أعمال وظيفته؟",
    "ما هي الحالات المحددة التي يبيح فيها القانون حق الدفاع الشرعي عن النفس حتى لو أدى ذلك إلى القتل العمد؟",
    "ما هي الشروط التي يجب توافرها لاعتبار الجاني عائداً، وما هو الأثر المترتب على ذلك عند تقدير العقوبة؟",

    # قانون المرافعات المدنية والتجارية
    "ما هو ميعاد الاستئناف المحدد للأحكام في الحالات العادية؟",
    "ما هي الشروط التي يجب توافرها في المصلحة لكي يتم قبول أي دعوى أو طلب أو دفع أمام القضاء؟",
    "إلى أي درجة من القرابة أو المصاهرة يعتبر القاضي غير صالح لنظر الدعوى؟",
    "ما هي الإجراءات الواجب اتباعها لإعلان الأوراق القضائية للأشخاص الذين لهم موطن معلوم في الخارج؟",
    "في أي حالات يعتبر القاضي غير صالح لنظر الدعوى ويمنع من سماعها حتى لو لم يرده الخصوم؟",
]

test_cases = []
results_data = []

# for question in questions:
#     print(f"\n{'='*60}")
#     print(f"Question: {question}")
#     print('='*60)

#     result = app.invoke({"question": question})
#     documents = result.get("documents") or []

#     with open("rag_agent\\tests\\retrieval_debug_log.txt", "a", encoding="utf-8") as log_file:
#         log_file.write(f"Q: {question}\n")
#         log_file.write(f"A: {result['answer']}\n")
#         log_file.write("Docs:\n")
#         for i, doc in enumerate(documents, start=1):
#             log_file.write(f"\n[Doc {i}]\n")
#             log_file.write(f"{doc.page_content.strip()}\n")
#             log_file.write("-"*20 + "\n")
#         log_file.write("\n" + "="*40 + "\n\n")

#     # print("\nAnswer:", result["answer"])



#     test_case = LLMTestCase(
#         input = question,
#         actual_output = result["answer"],
#         retrieval_context=[doc.page_content for doc in documents]
#     )

#     results_data.append({
#         "question": question,
#         "answer": result["answer"],
#         "retrieval_context": [doc.page_content for doc in documents]
#     })

#     test_cases.append(test_case)



# with open("rag_agent\\tests\\results.json", "w", encoding="utf-8") as f:
#     json.dump(results_data, f, ensure_ascii=False, indent=2)



with open("rag_agent\\tests\\results.json", "r", encoding="utf-8") as f:
    results_data = json.load(f)

for item in results_data:
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=item["answer"],
        retrieval_context=item["retrieval_context"]
    )
    test_cases.append(test_case)

print(f"Loaded {len(test_cases)} test cases from file")

llm = OllamaModel(
    model='gemma3:4b',
    temperature = 0
)



faithfulness_metric = FaithfulnessMetric(
    model = llm, 
    include_reason = False,
)

contextual_relevancy_metric = ContextualRelevancyMetric(
    model = llm,
    include_reason = False,
)

evaluate(test_cases = test_cases, metrics=[faithfulness_metric, contextual_relevancy_metric], async_config=AsyncConfig(run_async=False))



