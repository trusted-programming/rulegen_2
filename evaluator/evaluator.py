import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score, precision_score, recall_score

class Evaluator:
    def __init__(self, metric):
        self.metric = metric

    def compute(self, hypotheses, references):
        if self.metric == "bleu":
            return sacrebleu.corpus_bleu(hypotheses, [references]).score
        elif self.metric == "f1":
            return self.compute_f1_score(hypotheses, references)
        elif self.metric == "precision":
            return self.compute_precision_score(hypotheses, references)
        elif self.metric == "recall":
            return self.compute_recall_score(hypotheses, references)
        elif self.metric == "rouge":
            return self.compute_rouge(hypotheses, references)
        elif self.metric == "meteor":
            return self.compute_meteor(hypotheses, references)
        else:
            raise ValueError("Invalid metric specified")

    def compute_f1_score(self, hypotheses, references):
        # Calculate F1 score for your use case, this is just a sample
        return f1_score(hypotheses, references, average='weighted')

    def compute_precision_score(self, hypotheses, references):
        # Calculate precision score for your use case, this is just a sample
        return precision_score(hypotheses, references, average='weighted')

    def compute_recall_score(self, hypotheses, references):
        # Calculate recall score for your use case, this is just a sample
        return recall_score(hypotheses, references, average='weighted')

    def compute_rouge(self, hypotheses, references):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        rouge1 = sum([score['rouge1'].fmeasure for score in scores]) / len(scores)
        rougeL = sum([score['rougeL'].fmeasure for score in scores]) / len(scores)
        return {"rouge1": rouge1, "rougeL": rougeL}

    def compute_meteor(self, hypotheses, references):
        return sum([meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)]) / len(hypotheses)
