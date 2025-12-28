from _future_ import annotations

from dataclasses import dataclass
from typing import List
import requests


@dataclass
class SearchItem:
    title: str
    snippet: str
    link: str

    @property
    def combined_text(self) -> str:
        return f"{self.title} {self.snippet}".strip()


def google_custom_search(query: str, api_key: str, cx: str, num: int = 10) -> List[SearchItem]:
    q = query.replace(" ", "+")
    url = f"https://www.googleapis.com/customsearch/v1?q={q}&cx={cx}&key={api_key}&num={num}"

    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items", []) or []
    out: List[SearchItem] = []
    for it in items:
        title = it.get("title", "") or ""
        snippet = it.get("snippet", "") or ""
        link = it.get("link", "") or ""
        if (title + snippet).strip():
            out.append(SearchItem(title=title, snippet=snippet, link=link))
    return out