"""
src/graph/router.py  –  Intent classifier.

Intents: faq | troubleshoot | policy | chitchat

Uses keyword rules first (0ms), falls back to LLM only when ambiguous.
Chitchat queries skip ALL retrieval and reranking — huge latency saving.
"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.config import get_settings

Intent = Literal["faq", "troubleshoot", "policy", "chitchat"]
_settings = get_settings()

_CHITCHAT_RE = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|bye|goodbye|good morning|"
    r"good afternoon|how are you|what'?s up|yo|sup)\W*$",
    re.IGNORECASE,
)

_POLICY_KEYWORDS = {
    "refund", "cancel", "cancellation", "billing", "invoice", "subscription",
    "terms", "privacy", "gdpr", "payment", "charge", "price", "pricing",
    "plan", "upgrade", "downgrade", "data deletion",
}

_TROUBLESHOOT_KEYWORDS = {
    "error", "bug", "crash", "not working", "broken", "failed", "issue",
    "problem", "fix", "resolve", "500", "404", "exception", "traceback",
    "stuck", "timeout", "slow", "hang", "loop",
}


def _keyword_classify(query: str) -> Intent | None:
    if _CHITCHAT_RE.match(query.strip()):
        return "chitchat"
    ql = query.lower()
    if any(kw in ql for kw in _POLICY_KEYWORDS):
        return "policy"
    if any(kw in ql for kw in _TROUBLESHOOT_KEYWORDS):
        return "troubleshoot"
    return None


_ROUTER_PROMPT = """\
Classify the customer support query into one of these intents:
  faq          – general how-to or feature questions
  troubleshoot – errors, crashes, bugs, broken functionality
  policy       – billing, refunds, cancellation, terms, privacy
  chitchat     – greetings, thanks, off-topic, small talk
Reply with ONLY the single lowercase intent word."""


async def classify_intent(query: str) -> Intent:
    fast = _keyword_classify(query)
    if fast is not None:
        logger.debug(f"Intent (keyword): {fast}")
        return fast

    llm = ChatOpenAI(
        model=_settings.openai_model,
        temperature=0.0,
        api_key=_settings.openai_api_key,
        max_tokens=5,
    )
    try:
        resp = await llm.ainvoke([
            SystemMessage(content=_ROUTER_PROMPT),
            HumanMessage(content=query),
        ])
        intent_str = resp.content.strip().lower().split()[0]
        valid = {"faq", "troubleshoot", "policy", "chitchat"}
        result: Intent = intent_str if intent_str in valid else "faq"  # type: ignore
        logger.debug(f"Intent (LLM): {result}")
        return result
    except Exception as exc:
        logger.warning(f"Router LLM failed: {exc}. Default: faq")
        return "faq"
