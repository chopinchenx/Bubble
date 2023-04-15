from typing import List

from duckduckgo_search import ddg_news


def get_news(keywords: str) -> List[dict]:
    return ddg_news(keywords, safesearch="moderate", time='w', max_results=5)
