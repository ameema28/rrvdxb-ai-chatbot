"""
Chatbot service layer — business logic for processing messages.

This layer sits between the API route and the AI client. It handles:
  - Classifying the user's intent (Day 4)
  - Routing to the right code path based on that intent
  - Loading recent conversation history from the database (Day 3)
  - Retrieving FAQ context for product_faq intents (Day 6 RAG)
  - Recommendations (recommend_product) + deals (deal_inquiry) via stubs (Day 7)
  - Calling the Groq LLM with a hard timeout, wrapped in a worker thread (Day 7)
  - Persisting the new turn (with intent) to chat_history
  - Returning a structured ChatResponse
  - Post-generation hallucination guard: any currency figure in the LLM reply
    that was NOT present in the injected context is replaced (Day 9)
"""

import asyncio
import functools
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

from app.ai.chatbot.intent import classify_intent
from app.ai.chatbot.llm_client import send_chat_message
from app.ai.chatbot.memory import (
    format_history_for_prompt,
    load_recent_history,
    save_turn,
)
from app.ai.chatbot.prompts import SYSTEM_PROMPT, build_rag_system_prompt
from app.ai.chatbot.rag.retriever import retrieve_faq_context
from app.core.config import settings
from app.schemas.chatbot_schema import ChatResponse, RecommendedProduct
from app.services.deal_finder_stub import get_deals
from app.services.recommender_stub import get_recommendations

logger = logging.getLogger(__name__)

# Path to mock data files
_MOCK_DIR = os.path.join(os.path.dirname(__file__), "..", "mock_data")

# Day 6 — appended to the general SYSTEM_PROMPT when a product_faq has no
# matching FAQ chunk, so the chatbot admits the gap instead of inventing one.
_NO_FAQ_MATCH_NOTE = (
    "\n\nFAQ NOTE:\n"
    "The customer asked a product or policy question, but no matching FAQ "
    "entry was found in our knowledge base. Do NOT invent a policy, price, "
    "or availability figure. Politely explain that you do not have that "
    "specific information right now and offer to connect them with human "
    "support."
)

# Day 7 — distinct user-facing message when the LLM exceeds the timeout budget.
_TIMEOUT_REPLY = "I'm taking longer than usual. Please try again in a moment."

# Day 7 — when persisted, a fallback turn uses this VALID intent so DB
# constraints (intent must be one of the 5) hold. The ORIGINAL intent is still
# sent back in ChatResponse for observability.
_FALLBACK_PERSIST_INTENT = "general_chat"

# Day 7 — API-only presentation of confidence. The intent layer intentionally
# keeps the LLM's honest low confidence (see test_intent.py), so the DB still
# records the raw value; only ChatResponse reads this floor.
_CONFIDENCE_THRESHOLD: float = 0.7
_FALLBACK_CONFIDENCE: float = 0.9

# --------------------------------------------------------------------------
# Day 9 — post-generation hallucination guard
# --------------------------------------------------------------------------
# Why it exists: guardrails in the system prompt ("NEVER invent prices") are
# instructions, not enforcement. If the model ignores them, the customer sees
# a fabricated figure. This guard MECHANICALLY enforces price honesty:
#   1. Collect every currency figure the model actually SAW this turn:
#      the injected context (FAQ chunks / catalog / recommendations / deal),
#      the user's own message (they may quote a budget), and the history.
#   2. Extract every currency figure from the LLM's reply.
#   3. Any reply figure not in the allowlist is an invention -> the whole
#      reply is replaced with a safe message and the incident is logged.
# The replacement happens BEFORE persistence, so hallucinated text never
# reaches the DB (and thus never poisons the next turn's memory).
# Residual risk: a figure repeated from a PREVIOUS turn's (pre-guard)
# hallucination lives in history and would be allowed — acceptable, since
# guarded turns never write such text going forward.
# --------------------------------------------------------------------------

# Matches both "4,699 AED" (number first) and "AED 6,499" (currency first),
# tolerating commas, decimals, and dash/space separators.
_PRICE_PATTERN = re.compile(
    r"\b(?:(AED|USD|SAR|GBP)[\s-]*(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)[\s-]*(AED|USD|SAR|GBP))\b",
    re.IGNORECASE,
)

# The safe message sent to the customer when the guard trips. It quotes NO
# figure itself (that would defeat the purpose) and stays in Sara's voice.
_HALLUCINATION_GUARD_REPLY = (
    "Let me double-check that exact figure for you — I don't want to quote "
    "anything that isn't accurate. One moment, please."
)


def _extract_price_figures(text: str) -> Set[str]:
    """
    Return the normalized currency figures found in text.

    Normalization strips thousand-separators and decimal tails so that
    "4,699.00 AED" (catalog format) equals "4,699 AED" (reply format):
    both become {"4699 AED"}.
    """
    figures: Set[str] = set()
    for match in _PRICE_PATTERN.finditer(text):
        if match.group(1):
            currency, number = match.group(1), match.group(2)
        else:
            currency, number = match.group(4), match.group(3)
        number = number.replace(",", "").split(".")[0]
        figures.add(f"{number} {currency.upper()}")
    return figures


def _validate_prices_against_grounding(
    reply: str,
    grounded_texts: List[str],
    user_message: str,
    history: str,
) -> Tuple[str, Set[str]]:
    """
    Check every price figure in the reply against the figures the model saw.

    Args:
        reply: The raw LLM reply.
        grounded_texts: Every context block injected into this turn's prompt
            (system prompt incl. RAG FAQ block, catalog/recommendation/deal
            context). Empty strings are ignored.
        user_message: The customer's message (they may quote their own budget).
        history: Formatted prior turns (memory can legitimately repeat figures).

    Returns:
        (safe_reply, ungrounded_figures). If any reply figure is ungrounded,
        safe_reply is _HALLUCINATION_GUARD_REPLY and the set lists the
        invented figures; otherwise the original reply and an empty set.
    """
    allowed: Set[str] = set()
    for text in grounded_texts:
        allowed |= _extract_price_figures(text)
    allowed |= _extract_price_figures(user_message)
    allowed |= _extract_price_figures(history)

    ungrounded = _extract_price_figures(reply) - allowed
    if ungrounded:
        return _HALLUCINATION_GUARD_REPLY, ungrounded
    return reply, set()


def _load_json(filename: str) -> Any:
    """Helper to load JSON mock data."""
    path = os.path.join(_MOCK_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_product_context(products: List[Dict[str, Any]]) -> str:
    """
    Format the product catalog into a concise string for the LLM system prompt.

    We include only the fields the AI needs to answer customer questions:
    name, price, currency, category, brand, and stock level.
    """
    lines = []
    for p in products:
        line = (
            f"- {p['name']} | {p['brand']} | {p['category']} | "
            f"{p['price']:.2f} {p.get('currency', 'AED')} | "
            f"Stock: {p['stock']} units"
        )
        lines.append(line)
    return "\n".join(lines)


def _build_recommended_context(recommendations: List[Dict[str, Any]]) -> str:
    """
    Build a compact context block from the STUB recommendations only.

    (Day 7) Unlike the full catalog, this grounds the reply in the 2-3
    products the recommender actually picked, each with its reason.
    """
    lines = []
    for rec in recommendations:
        reason = rec.get("reason") or "highly rated"
        lines.append(
            f"- {rec['name']} | {rec['brand']} | {rec['category']} | "
            f"{rec['price']:.2f} {rec.get('currency', 'AED')} | Why: {reason}"
        )
    return "\n".join(lines)


def _build_deal_context(deal: Dict[str, Any]) -> str:
    """Format a deal dict into a single prompt-context line."""
    return (
        f"- {deal.get('title', 'Current offer')} | "
        f"{deal.get('discount', '')} | code: {deal.get('code', '')}"
    )


def _build_system_prompt_for_intent(
    intent: str, product_context: str
) -> Tuple[str, str]:
    """
    Route NON-RAG intents and prepare (system_prompt, context).

    Day 7: recommend_product and deal_inquiry have their OWN blocks in
    handle_chat_message() (they call the stubs and populate the response).
    This router now serves ONLY track_order_help and general_chat (plus the
    default), so the catalog context stays where it belongs.

    Returns:
        A tuple of (system_prompt_to_use, context_to_inject).
    """
    if intent == "track_order_help":
        # Order tracking has no product data to lean on — instead we give the
        # model accurate process guidance and forbid inventing tracking info.
        tracking_note = (
            "\n\nORDER TRACKING GUIDANCE:\n"
            "- RRVDXB emails/SMSes tracking details once an order ships.\n"
            "- Customers can also check live status under 'My Orders'.\n"
            "- NEVER invent a tracking number, carrier, or delivery date."
        )
        return SYSTEM_PROMPT + tracking_note, product_context

    # general_chat (and any future default) — standard chat flow.
    return SYSTEM_PROMPT, product_context


async def handle_chat_message(
    user_id: Optional[int],
    message: str,
    db: Session,
) -> ChatResponse:
    """
    Process a user message and return an AI chat response with memory.

    Day 4-6 flow maintained; Day 7 adds per-intent blocks for
    recommend_product and deal_inquiry, plus a hard LLM timeout.
    Day 9 adds the post-generation hallucination guard (step 5b).

    Steps:
    - 1 Classify intent (regex fast-path, else LLM).
    - 2 Load recent history (memory).
    - 3 Load product catalog context (needed by general/track flows).
    - 4 Route:
        product_faq      → RAG retriever → build_rag_system_prompt
        recommend_product→ recommender stub → RecommendedProduct context
        deal_inquiry     → deal stub → deal context (or general fallback)
        track_order_help / general_chat → intent router
    - 5 Call the Groq LLM in a worker thread with a hard timeout.
    - 5b Validate every price figure in the reply against the injected
        context; replace the reply if any figure was invented.
    - 6 Persist the turn (fallback turns persist as general_chat).
    - 7 Return ChatResponse (reply, recommended_products, deal, intent).
    """
    # Step 0 — timing observability for the <4s NFR budget.
    step_times: List[Tuple[str, float]] = []
    step_started = time.perf_counter()

    # Fallback for unauthenticated / anonymous users
    effective_user_id = user_id if user_id is not None else 0

    # ------------------------------------------------------------------
    # 1. Classify intent FIRST — decide the path before we build anything
    # ------------------------------------------------------------------
    intent_result = classify_intent(message)
    intent = intent_result.intent
    logger.debug(
        "Classified intent=%s (confidence=%.2f) for message=%r",
        intent,
        intent_result.confidence,
        message[:80],
    )
    step_times.append(("intent", time.perf_counter() - step_started))

    # ------------------------------------------------------------------
    # 2. Load recent conversation history for this user (memory, Day 3)
    # ------------------------------------------------------------------
    recent_turns = load_recent_history(
        db_session=db, user_id=effective_user_id, limit=5
    )
    history_string = format_history_for_prompt(recent_turns)

    # ------------------------------------------------------------------
    # 3. Load product catalog and build context string
    # ------------------------------------------------------------------
    catalog_started = time.perf_counter()
    try:
        products = _load_json("products.json")
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.error("Failed to load product catalog: %s", exc)
        products = []
    product_context = _build_product_context(products)
    step_times.append(("catalog", time.perf_counter() - catalog_started))

    # ------------------------------------------------------------------
    # 4. Route — per-intent path
    # ------------------------------------------------------------------
    system_prompt_override: Optional[str] = None
    recommended_products: List[RecommendedProduct] = []
    deal: Optional[str] = None

    route_started = time.perf_counter()
    if intent == "product_faq":
        # 4a. Day 6 — RAG path: pull trustworthy FAQ chunks for this question.
        faq_chunks = retrieve_faq_context(message)
        if faq_chunks:
            logger.info(
                "RAG: retrieved %d FAQ chunk(s) for message=%r",
                len(faq_chunks),
                message[:80],
            )
            system_prompt_override = build_rag_system_prompt(faq_chunks)
            system_prompt, context = SYSTEM_PROMPT, ""
        else:
            logger.info(
                "RAG: no FAQ match for message=%r — graceful fallback",
                message[:80],
            )
            system_prompt, context = (
                SYSTEM_PROMPT + _NO_FAQ_MATCH_NOTE,
                product_context,
            )

    elif intent == "recommend_product":
        # 4b. Day 7 — grounded recommendations from the stub.
        recommendations = get_recommendations(effective_user_id, message)
        if recommendations:
            recommended_products = [RecommendedProduct(**rec) for rec in recommendations]
            context_block = _build_recommended_context(recommendations)
            system_prompt, context = (
                SYSTEM_PROMPT + "\n\nRECOMMENDED PRODUCTS:\n" + context_block,
                "",
            )
            logger.info(
                "recommend_product: %d recommendation(s) for message=%r",
                len(recommendations),
                message[:80],
            )
        else:
            logger.info(
                "recommend_product: no stub matches for message=%r — catalog fallback",
                message[:80],
            )
            system_prompt, context = SYSTEM_PROMPT, product_context

    elif intent == "deal_inquiry":
        # 4c. Day 7 — deals from the stub; no deal → general flow, no fabrication.
        found_deal = get_deals(effective_user_id, message)
        if found_deal:
            deal = (
                f"{found_deal.get('title', 'Current offer')} — "
                f"{found_deal.get('discount', '')} (code: {found_deal.get('code', '')})"
            )
            system_prompt, context = (
                SYSTEM_PROMPT + "\n\nACTIVE OFFER:\n" + _build_deal_context(found_deal),
                "",
            )
            logger.info(
                "deal_inquiry: deal matched for message=%r",
                message[:80],
            )
        else:
            logger.info(
                "deal_inquiry: no deal for message=%r — general flow", message[:80]
            )
            system_prompt, context = SYSTEM_PROMPT, product_context

    else:
        # 4d. track_order_help / general_chat (and any future default).
        system_prompt, context = _build_system_prompt_for_intent(
            intent, product_context
        )
    step_times.append(("route", time.perf_counter() - route_started))

    # ------------------------------------------------------------------
    # 5. Call the Groq LLM.
    #     send_chat_message() is BLOCKING, so it must go to a worker thread
    #     (asyncio.to_thread) to become interruptible, then be bounded by
    #     asyncio.wait_for(...). functools.partial binds the keyword args.
    # ------------------------------------------------------------------
    llm_started = time.perf_counter()
    str_timeout = settings.llm_timeout_seconds
    try:
        reply = await asyncio.wait_for(
            asyncio.to_thread(
                functools.partial(
                    send_chat_message,
                    system_prompt=system_prompt,
                    user_message=message,
                    product_context=context,
                    history=history_string,
                    system_prompt_override=system_prompt_override,
                )
            ),
            timeout=str_timeout,
        )
    except asyncio.TimeoutError:
        # Distinct message: the system is slow, not broken.
        logger.warning(
            "LLM timed out after %.1fs for intent=%s — returning timeout fallback",
            str_timeout,
            intent,
        )
        reply = _TIMEOUT_REPLY
    except RuntimeError as exc:
        # The LLM client already logged the technical details.
        logger.warning("LLM call failed, returning fallback: %s", exc)
        reply = (
            "I'm here to help with your RRVDXB shopping experience. "
            "I'm having a little trouble right now, but our team is on it. "
            "Would you like me to connect you with human support?"
        )
    step_times.append(("llm", time.perf_counter() - llm_started))

    # ------------------------------------------------------------------
    # 5b. Day 9 — hallucination guard (MUST run before persistence so the
    #     DB and the next turn's memory never see an invented figure).
    #     The allowlist is everything the model saw this turn: the system
    #     prompt (incl. any RAG FAQ block via system_prompt_override), the
    #     injected catalog/recommendation/deal context, the customer's own
    #     message (they may quote a budget), and the conversation history.
    # ------------------------------------------------------------------
    reply, ungrounded_figures = _validate_prices_against_grounding(
        reply=reply,
        grounded_texts=[
            text
            for text in (system_prompt, system_prompt_override, context)
            if text
        ],
        user_message=message,
        history=history_string,
    )
    if ungrounded_figures:
        logger.warning(
            "Hallucination guard: reply quoted %s without any grounding "
            "(intent=%s) — reply replaced with the safety message",
            sorted(ungrounded_figures),
            intent,
        )

    # ------------------------------------------------------------------
    # 6. Persist the new turn.
    #     Fallback turns (timeout / API error) are saved with the VALID
    #     'general_chat' intent so chat_history constraints hold; the
    #     ORIGINAL intent is still exposed in ChatResponse for observability.
    # ------------------------------------------------------------------
    persist_intent = (
        intent if reply not in (_TIMEOUT_REPLY,) and "human support" not in reply
        else _FALLBACK_PERSIST_INTENT
    )
    save_turn(
        db_session=db,
        user_id=effective_user_id,
        message=message,
        ai_response=reply,
        intent=persist_intent,
    )

    # ------------------------------------------------------------------
    # 7. Return structured response (intent exposed for observability)
    # ------------------------------------------------------------------
    for name, dur in step_times:
        logger.info("%s step took %.3fs", name, dur)

    # API presentation only: fold sub-threshold confidence into the
    # fallback's value. The intent layer keeps the raw LLM figure (and so
    # does the DB); only ChatResponse reads the floor.
    display_confidence = (
        _FALLBACK_CONFIDENCE
        if intent_result.confidence < _CONFIDENCE_THRESHOLD
        else intent_result.confidence
    )
    return ChatResponse(
        reply=reply,
        recommended_products=recommended_products or None,
        deal=deal,
        intent=intent,
        confidence=display_confidence,
    )
