from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.schemas import SampleHeadline, SentimentResult

MODEL_NAME = "VADER (vaderSentiment)"

# Path to the local Financial PhraseBank file (sentence@label format)
PHRASEBANK_PATH = Path(__file__).parent.parent / "FinancialPhraseBank" / "Sentences_AllAgree.txt"

# Thresholds per VADER docs
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def _load_phrasebank(path: Path) -> list[SampleHeadline]:
    samples = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            label = label.strip().lower()
            if label in ("positive", "negative", "neutral"):
                samples.append(SampleHeadline(sentence=sentence.strip(), label=label))
    return samples


class VADERAnalyzer:
    def __init__(self):
        print("Loading VADER sentiment analyzer...")
        self._analyzer = SentimentIntensityAnalyzer()
        print("VADER loaded.")

        print(f"Loading Financial PhraseBank from {PHRASEBANK_PATH}...")
        self._samples = _load_phrasebank(PHRASEBANK_PATH)
        print(f"Dataset loaded: {len(self._samples)} samples.")

    def analyze(self, headline: str) -> SentimentResult:
        scores = self._analyzer.polarity_scores(headline)
        compound = scores["compound"]

        if compound >= POSITIVE_THRESHOLD:
            sentiment = "positive"
        elif compound <= NEGATIVE_THRESHOLD:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        class_scores = {
            "positive": round(scores["pos"], 4),
            "negative": round(scores["neg"], 4),
            "neutral":  round(scores["neu"], 4),
        }

        return SentimentResult(
            headline=headline,
            sentiment=sentiment,
            confidence=round(class_scores[sentiment], 4),
            scores=class_scores,
        )

    def analyze_batch(self, headlines: list[str]) -> list[SentimentResult]:
        return [self.analyze(h) for h in headlines]

    def get_samples(
        self, count: int = 10, sentiment: str | None = None
    ) -> list[SampleHeadline]:
        pool = self._samples
        if sentiment:
            pool = [s for s in pool if s.label == sentiment]
        return pool[:count]

    @property
    def sample_count(self) -> int:
        return len(self._samples)


# Module-level singleton; initialized in FastAPI lifespan
analyzer: VADERAnalyzer | None = None
