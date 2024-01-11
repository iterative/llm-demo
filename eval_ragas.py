import pandas as pd
# from datasets import load_dataset
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    # context_recall, # TODO: import context_recall once you have the ground_truths loaded correctly
    context_precision,
)
from ragas import evaluate


predictions = pd.read_csv("results.csv")
questions = predictions['Q'].to_list()
answers = predictions['A'].to_list()

# TODO: Load the correct values for contexts
contexts = [['DVC Doc 1', 'DVC Doc 2']] * len(answers)

# TODO: Load the correct values for ground_truths
# truth = pd.read_csv("canfy.csv")
# ground_truths = truth['A'].to_list()

dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        # "ground_truths": ground_truths # TODO: pass ground_truths once you load it correctly
        })

result = evaluate(
    dataset.select(range(1)), # TODO: Remove select to evaluate all samples
    metrics=[
        # context_precision, # TODO: Enable this metric. I've disabled it now coz it's taking a long time to run.
        # faithfulness, # TODO: Enable this metric. I've disabled it now coz it's taking a long time to run.
        answer_relevancy,
        # context_recall, # TODO: use context_recall once you have the ground_truths loaded correctly
    ],
)

df = result.to_pandas()
df.to_csv("eval_ragas.csv", header=True, index=False)
