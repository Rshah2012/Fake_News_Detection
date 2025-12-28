from __future__ import annotations
from __future__ import annotations
import requests
import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================
# CONFIG
# ============================
MODEL_NAME = "roberta-large-mnli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TAU_ENTAIL = 0.65
TAU_CONTRA = 0.65
SIMILARITY_THRESHOLD = 0.15  # relevance filter
TOP_K = 10


# ============================
# INPUT
# ============================
user_text = input("Enter headline: ")

query = user_text.replace(" ", "+")
url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}"
response = requests.get(url).json()
items = response.get("items", [])

search_results = []
for item in items:
    combined = f"{item.get('title','')} {item.get('snippet','')}"
    if combined.strip():
        search_results.append(combined)

if not search_results:
    print("FINAL VERDICT: NOT ENOUGH INFO")
    exit()


# ============================
# UTILITIES
# ============================
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def relevance_filter(claim: str, articles: List[str]) -> List[str]:
    vec = TfidfVectorizer(stop_words="english")
    tfidf = vec.fit_transform([claim] + articles)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()

    return [
        articles[i]
        for i in range(len(articles))
        if sims[i] >= SIMILARITY_THRESHOLD
    ]


# ============================
# NLI
# ============================
@dataclass
class NLIResult:
    article: str
    entailment: float
    contradiction: float
    neutral: float
    verdict: str


class NLIChecker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
        self.model.eval()
        self.labels = ["contradiction", "neutral", "entailment"]

    @torch.no_grad()
    def score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        enc = self.tokenizer(
            normalize(premise),
            normalize(hypothesis),
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        probs = F.softmax(self.model(**enc).logits, dim=-1)[0].cpu().tolist()
        return dict(zip(self.labels, probs))

    def bidirectional(self, article: str, claim: str) -> NLIResult:
        a1 = self.score(article, claim)
        a2 = self.score(claim, article)

        entail = max(a1["entailment"], a2["entailment"])
        contra = max(a1["contradiction"], a2["contradiction"])
        neutral = max(a1["neutral"], a2["neutral"])

        if contra >= TAU_CONTRA:
            verdict = "CONTRADICTS"
        elif entail >= TAU_ENTAIL:
            verdict = "SUPPORTS"
        else:
            verdict = "UNCLEAR"

        return NLIResult(article, entail, contra, neutral, verdict)


# ============================
# PIPELINE
# ============================
relevant_articles = relevance_filter(user_text, search_results)

checker = NLIChecker()
results = [checker.bidirectional(a, user_text) for a in relevant_articles]

supports = [r for r in results if r.verdict == "SUPPORTS"]
contradicts = [r for r in results if r.verdict == "CONTRADICTS"]
unclear = [r for r in results if r.verdict == "UNCLEAR"]

# ============================
# FINAL DECISION
# ============================
support_score = sum(r.entailment for r in supports)
contra_score = sum(r.contradiction for r in contradicts)

total_evidence = support_score + contra_score + 1e-6

confidence = abs(support_score - contra_score) / total_evidence

if contra_score > support_score * 1.2 and contra_score > 0.7:
    final_verdict = "FALSE"
elif support_score > contra_score * 1.2 and support_score > 0.7:
    final_verdict = "TRUE"
else:
    final_verdict = "NOT ENOUGH INFO"


# ============================
# OUTPUT
# ============================
print("\n==============================")
print("USER HEADLINE:")
print(user_text)

print("\n--- ARTICLE-LEVEL NLI RESULTS ---")
for r in results:
    print("\nARTICLE:")
    print(r.article)
    print(f"SUPPORT={r.entailment:.3f}  CONTRA={r.contradiction:.3f}  NEUTRAL={r.neutral:.3f}")
    print("VERDICT:", r.verdict)

print("\n==============================")
print("FINAL VERDICT:", final_verdict)
print(f"CONFIDENCE SCORE: {confidence:.2f}")
print("==============================")
