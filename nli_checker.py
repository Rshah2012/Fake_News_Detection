from _future_ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#MODEL_NAME = "roberta-large-mnli"
#MODEL_NAME = "microsoft/deberta-v3-base-mnli"
MODEL_NAME = "roberta-base-mnli"
LABELS_FALLBACK = ["contradiction", "neutral", "entailment"]

TAU_ENTAIL = 0.65
TAU_CONTRA = 0.65


def normalise_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class NLIResult:
    headline: str
    verdict: str  # MATCH_ENTAILS / MISMATCH_CONTRADICTS / UNCLEAR_NEUTRAL
    entailment: float
    contradiction: float
    neutral: float
    details: Dict[str, Any] | None = None


class NLIChecker:
    def _init_(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict) and len(id2label) == 3:
            mapped = [str(id2label[i]).lower() for i in range(3)]
            if set(mapped) == {"contradiction", "neutral", "entailment"}:
                self.labels = mapped
            else:
                self.labels = LABELS_FALLBACK
        else:
            self.labels = LABELS_FALLBACK

    @torch.no_grad()
    def scores(self, premise: str, hypothesis: str) -> Dict[str, float]:
        premise = normalise_text(premise)
        hypothesis = normalise_text(hypothesis)

        enc = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)

        logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().tolist()
        return dict(zip(self.labels, probs))

    def decision_one_way(
        self,
        premise: str,
        hypothesis: str,
        tau_entail: float = TAU_ENTAIL,
        tau_contra: float = TAU_CONTRA,
    ) -> Tuple[str, Dict[str, float]]:
        s = self.scores(premise, hypothesis)
        if s.get("entailment", 0.0) >= tau_entail:
            return "MATCH_ENTAILS", s
        if s.get("contradiction", 0.0) >= tau_contra:
            return "MISMATCH_CONTRADICTS", s
        return "UNCLEAR_NEUTRAL", s

    def decision_bidir(
        self,
        headline: str,
        user_input: str,
        tau_entail: float = TAU_ENTAIL,
        tau_contra: float = TAU_CONTRA,
    ) -> NLIResult:
        v1, s1 = self.decision_one_way(headline, user_input, tau_entail, tau_contra)
        v2, s2 = self.decision_one_way(user_input, headline, tau_entail, tau_contra)

        if s1.get("contradiction", 0.0) >= tau_contra or s2.get("contradiction", 0.0) >= tau_contra:
            verdict = "MISMATCH_CONTRADICTS"
        elif s1.get("entailment", 0.0) >= tau_entail or s2.get("entailment", 0.0) >= tau_entail:
            verdict = "MATCH_ENTAILS"
        else:
            verdict = "UNCLEAR_NEUTRAL"

        entail = max(s1.get("entailment", 0.0), s2.get("entailment", 0.0))
        contra = max(s1.get("contradiction", 0.0), s2.get("contradiction", 0.0))
        neutral = max(s1.get("neutral", 0.0), s2.get("neutral", 0.0))

        return NLIResult(
            headline=headline,
            verdict=verdict,
            entailment=float(entail),
            contradiction=float(contra),
            neutral=float(neutral),
            details={"headline->input": s1, "input->headline": s2, "verdicts": (v1, v2)},
        )


def check_input_against_headlines(
    user_input: str,
    headlines: List[str],
    top_k: int = 10,
    checker: NLIChecker | None = None,
) -> Dict[str, List[NLIResult]]:
    checker = checker or NLIChecker()

    results: List[NLIResult] = []
    for h in headlines:
        if isinstance(h, str) and h.strip():
            results.append(checker.decision_bidir(h, user_input))

    supports = [r for r in results if r.verdict == "MATCH_ENTAILS"]
    contradicts = [r for r in results if r.verdict == "MISMATCH_CONTRADICTS"]
    unclear = [r for r in results if r.verdict == "UNCLEAR_NEUTRAL"]

    supports.sort(key=lambda r: r.entailment, reverse=True)
    contradicts.sort(key=lambda r: r.contradiction, reverse=True)
    unclear.sort(key=lambda r: r.neutral, reverse=True)

    return {
        "supports": supports[:top_k],
        "contradicts": contradicts[:top_k],
        "unclear": unclear[:top_k],
    }