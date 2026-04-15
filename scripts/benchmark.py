"""
Benchmark VADER against the Financial PhraseBank (Sentences_AllAgree.txt).
2264 sentences with 100% annotator agreement.

Run: python -m scripts.benchmark
"""

from collections import Counter
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PHRASEBANK_PATH = Path(__file__).parent.parent / "FinancialPhraseBank" / "Sentences_AllAgree.txt"

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def load_data(path: Path) -> list[tuple[str, str]]:
    samples = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            label = label.strip().lower()
            if label in ("positive", "negative", "neutral"):
                samples.append((sentence.strip(), label))
    return samples


def predict(analyzer: SentimentIntensityAnalyzer, sentence: str) -> str:
    compound = analyzer.polarity_scores(sentence)["compound"]
    if compound >= POSITIVE_THRESHOLD:
        return "positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "negative"
    return "neutral"


def run_benchmark():
    print(f"Loading data from {PHRASEBANK_PATH}...")
    data = load_data(PHRASEBANK_PATH)
    print(f"Loaded {len(data)} samples.")

    analyzer = SentimentIntensityAnalyzer()

    true_labels = [label for _, label in data]
    pred_labels = [predict(analyzer, sentence) for sentence, _ in data]

    correct = sum(p == t for p, t in zip(pred_labels, true_labels))
    accuracy = correct / len(true_labels)
    print(f"\nAccuracy: {correct}/{len(true_labels)} = {accuracy:.2%}")

    classes = ["positive", "negative", "neutral"]
    true_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)

    print("\nPer-class results:")
    print(f"{'Class':<12} {'True':<8} {'Predicted':<10} {'Correct':<8} {'Precision':<10} {'Recall'}")
    print("-" * 62)
    for cls in classes:
        tp = sum(p == t == cls for p, t in zip(pred_labels, true_labels))
        precision = tp / pred_counts[cls] if pred_counts[cls] else 0.0
        recall = tp / true_counts[cls] if true_counts[cls] else 0.0
        print(
            f"{cls:<12} {true_counts[cls]:<8} {pred_counts[cls]:<10} "
            f"{tp:<8} {precision:<10.2%} {recall:.2%}"
        )

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(f"{'':>12}", "  ".join(f"{c:>10}" for c in classes))
    for true_cls in classes:
        row = [
            sum(p == pred_cls and t == true_cls for p, t in zip(pred_labels, true_labels))
            for pred_cls in classes
        ]
        print(f"{true_cls:>12}", "  ".join(f"{v:>10}" for v in row))


if __name__ == "__main__":
    run_benchmark()
