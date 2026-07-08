import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.models import OpenRouterModel

from rag_agent.agent import build_graph
import json
from deepeval.evaluate import AsyncConfig



app = build_graph()

questions = [
    # القانون المدني (قانون رقم 131 لسنة 1948)
    "ما هو التسلسل القانوني للمصادر التي يجب على القاضي تطبيقها في حال انعدام النص التشريعي؟",
    "تحت أي ظروف يُعتبر استعمال الشخص لحقه غير مشروع وفقاً للمادة 5؟",
    "كيف يتم حساب درجة القرابة المباشرة ودرجة قرابة الحواشي؟",
    "ما هو السن القانوني الذي حدده القانون لبلوغ سن الرشد واكتساب الأهلية الكاملة لمباشرة الحقوق المدنية؟",
    "ما هو المعيار القانوني الذي وضعه المشرع للتمييز بين العقار والمنقول؟",

    # قانون الإجراءات الجنائية (قانون رقم 150 لسنة 1950)
    "من هي الجهة التي يختص القانون بمنحها دون غيرها سلطة رفع الدعوى الجنائية ومباشرتها؟",
    "ما هو الميعاد المحدد لمأمور الضبط القضائي لإرسال المتهم المقبوض عليه إلى النيابة العامة، وما هي المدة الممنوحة للنيابة لاستجوابه؟",
    "ما هي الحالات التي يتوقف فيها رفع الدعوى الجنائية على تقديم شكوى شفهية أو كتابية من المجني عليه؟",
    "ما هي المدد الزمنية القصوى للحبس الاحتياطي التي يجوز لقاضي التحقيق أن يأمر بها قبل الإحالة لغرفة المشورة؟",
    "ما هي الحالات الخمس المحددة قانوناً التي يجوز فيها طلب إعادة النظر في الأحكام النهائية الصادرة بالعقوبة؟",

    # قانون التجارة (قانون رقم 17 لسنة 1999)
    "ما هو التسلسل القانوني للقواعد والعادات الواجبة التطبيق على المواد التجارية في حال عدم وجود اتفاق؟",
    "ما هو الحد الأدنى لرأس المال المستثمر الذي يلزم التاجر بإمساك دفاتر تجارية منتظمة؟",
    "ما هي العناصر التي يتألف منها المتجر كمال منقول مخصص لمزاولة تجارة معينة؟",
    "ما هي البيانات الستة الأساسية التي يجب أن يشتمل عليها الشيك ليعتبر صحيحاً من الناحية القانونية؟",
    "ما هي الأفعال التي اعتبرها القانون جرائم شيك تستوجب عقوبة الحبس والغرامة المالية؟",

    # قانون العقوبات (قانون رقم 58 لسنة 1937)
    "كيف يتم التمييز بين الجنايات والجنح والمخالفات بناءً على نوع العقوبات المقررة لكل منها؟",
    "ما هو التعريف القانوني لـ الشروع في الجريمة، وهل يُعاقب القانون على مجرد العزم أو الأعمال التحضيرية؟",
    "ما المقصود بـ الإصرار السابق و الترصد كظروف مشددة في جرائم القتل العمد؟",
    "من هم الأشخاص الذين يُعدون في حكم الموظف العام في تطبيق نصوص فصل الرشوة؟",
    "كيف عرف القانون جريمة الإرهاب وما هي الغايات التي يهدف إليها الجاني من خلالها؟",

    # قانون المرافعات المدنية والتجارية (قانون رقم 13 لسنة 1968)
    "من هو الشخص الذي يمثل قاضي الأمور الوقتية في محكمة المواد الجزئية؟",
    "ما هي البيانات الإلزامية التي يجب أن تشتمل عليها الأوراق التي يقوم المحضرون بإعلانها؟",
    "ما هو النصاب المالي الذي يحدد الاختصاص النوعي لمحكمة المواد الجزئية في الدعاوى المدنية والتجارية؟",
    "ما هي الشروط الواجب توافرها في حق الدائن لكي يتبع طريق أمر الأداء بدلاً من القواعد العامة لرفع الدعاوى؟",
    "ما هي الحالات المحددة التي يجوز فيها للخصوم الطعن أمام محكمة النقض في الأحكام الصادرة من محاكم الاستئناف؟",
]

test_cases = []
results_data = []

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    result = app.invoke({"question": question})
    documents = result.get("documents") or []

    with open("rag_agent\\tests\\retrieval_debug_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"Q: {question}\n")
        log_file.write(f"A: {result['answer']}\n")
        log_file.write("Docs:\n")
        for doc in documents:
            log_file.write(f"- {doc.page_content.strip()}\n")
        log_file.write("\n" + "-"*40 + "\n\n")

        print("\nAnswer:", result["answer"])



    test_case = LLMTestCase(
        input = question,
        actual_output = result["answer"],
        retrieval_context=[doc.page_content for doc in documents]
    )

    results_data.append({
        "question": question,
        "answer": result["answer"],
        "retrieval_context": [doc.page_content for doc in documents]
    })

    test_cases.append(test_case)



# with open("rag_agent\\tests\\results.json", "w", encoding="utf-8") as f:
#     json.dump(results_data, f, ensure_ascii=False, indent=2)



# with open("rag_agent\\tests\\results.json", "r", encoding="utf-8") as f:
#     results_data = json.load(f)

# for item in results_data:
#     test_case = LLMTestCase(
#         input=item["question"],
#         actual_output=item["answer"],
#         retrieval_context=item["retrieval_context"]
#     )
#     test_cases.append(test_case)

# print(f"Loaded {len(test_cases)} test cases from file")

# llm = OllamaModel(
#     model='qwen2.5:3b-instruct',
#     temperature = 0
# )

llm = OpenRouterModel(
    model="",
    api_key="",
    temperature=0,
)

faithfulness_metric = FaithfulnessMetric(
    model = llm, 
    include_reason = False,
    async_mode = False,
    # threshold = 0.6

)
contextual_relevancy_metric = ContextualRelevancyMetric(
    model = llm,
    include_reason = False,
    async_mode = False,
    # threshold=0.6

)

evaluate(test_cases = test_cases, metrics=[faithfulness_metric, contextual_relevancy_metric], async_config=AsyncConfig(run_async=False))

# evaluate(test_cases = test_cases, metrics=[faithfulness_metric, ], async_config=AsyncConfig(run_async=False))
# evaluate(test_cases = test_cases, metrics=[ contextual_relevancy_metric], async_config=AsyncConfig(run_async=False))


