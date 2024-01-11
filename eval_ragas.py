import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from ragas import evaluate


predictions = pd.read_csv("results.csv")
questions = predictions['Q'].to_list()
answers = predictions['A'].to_list()

# TODO: Load the correct values for contexts
contexts = [['DVC Doc 1', 'DVC Doc 2']] * len(answers)

truth = pd.read_csv("canfy.csv")
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
df.to_csv("eval_ragas.csv", header=True, index=False)
