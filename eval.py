import json
import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate


with open("results.json") as f:
    results = json.load(f)

questions, answers, contexts = [], [], []
for result in results:
    questions.append(result['Q'])
    answers.append(result['A'])
    contexts.append(result['context'])

truth = pd.read_csv("ground_truths.csv")
ground_truths = truth['A'].to_list()
# Convert to provide a list of ground_truths for each question.
ground_truths = [[ground_truth] for ground_truth in ground_truths]

dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
        })

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)

df = result.to_pandas()
df.to_csv("eval.csv", header=True, index=False)
