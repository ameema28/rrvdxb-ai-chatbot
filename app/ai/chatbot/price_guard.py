"""
Post-generation price guard (deterministic, no LLM).

Why it exists: prompt rules ("NEVER invent prices") are instructions, not
enforcement. This guard checks every currency figure in the LLM reply and,
if one fails, the WHOLE reply is replaced with a safe message before it is
returned or saved (so it never reaches the DB or later memory).

Two checks, both must pass (AND):

1. Grounding (presence): every figure in the reply must appear in what the
   model saw this turn: the system prompt / RAG block / injected context,
   the user's own message, or the conversation history.

2. Per-product (heuristic): if a figure is attached to a named catalog
   product, it must be THAT product's catalog price and currency. This
   catches a real catalog price attached to the wrong product, which the
   presence check alone lets through whenever the catalog is injected.

   - Product mentions: case-insensitive, whole-word matches of each
     product's name and its explicit `aliases` (products.json).
   - The reply is split into clauses at newlines, bullets, ";" and sentence
     ends (periods inside a price token, e.g. "Rs. 4,699", do not split).
   - A figure is bound to the nearest product mention BEFORE it in the same
     clause, unless one of the 3 words just before the figure is a
     qualifier (under, over, below, above, up, less, more, than, budget,
     from), as in "gifts under 500 AED".

Currency forms recognised (code/symbol before or after the number, commas
and decimals allowed): AED, USD, SAR, GBP, PKR, INR, "Rs", "Rs.", the rupee
signs, "$", the pound sign, "dirham"/"dirhams". Everything is normalised to
an ISO code: $ -> USD, pound sign -> GBP, Rs / Rs. / small rupee sign ->
PKR, Indian rupee sign -> INR, dirham(s) -> AED. Amounts are compared
exactly to the cent (4,699.00 == 4699, but 4699.50 != 4699).

Known limits (this is a heuristic, not a proof of correctness):
  - Bare numbers without a currency are not checked.
  - A figure written BEFORE the product name ("for 899 AED you get the
    iPhone 14 Pro Max") is not bound; only the presence check applies.
  - Paraphrased product references ("Apple's flagship phone") are not
    detected; aliases must be maintained in products.json.
  - Computed figures (discounted prices, currency conversions) are not in
    the catalog, so replies quoting them are replaced (safe direction).
  - Only price figures are guarded; other facts (stock, delivery times,
    policies) are not checked here.
"""

import bisect
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# The safe message sent when the guard trips. It quotes no figure itself.
GUARD_REPLY = (
    "Let me double-check that exact figure for you — I don't want to quote "
    "anything that isn't accurate. One moment, please."
)

_CURRENCY_ALIASES: Dict[str, str] = {
    "aed": "AED",
    "usd": "USD",
    "sar": "SAR",
    "gbp": "GBP",
    "pkr": "PKR",
    "inr": "INR",
    "rs": "PKR",
    "rs.": "PKR",
    "dirham": "AED",
    "dirhams": "AED",
    "$": "USD",
    "£": "GBP",  # pound sign
    "₨": "PKR",  # rupee sign (used in Pakistan)
    "₹": "INR",  # Indian rupee sign
}

_WORD_CURRENCY = r"AED|USD|SAR|GBP|PKR|INR|Rs\.?|dirhams?"
_SYMBOL_CURRENCY = "[$£₨₹]"
_NUMBER = r"\d[\d,]*(?:\.\d+)?"

_PRICE_PATTERN = re.compile(
    # currency first: "AED 6,499", "Rs. 4699", "$1,299.50"
    rf"(?:(?<![A-Za-z])(?P<pre_word>{_WORD_CURRENCY})(?![A-Za-z])"
    rf"|(?P<pre_sym>{_SYMBOL_CURRENCY}))[\s-]*(?P<n1>{_NUMBER})"
    # number first: "4,699 AED", "489 dirhams", "50£"
    rf"|(?<![\w.,])(?P<n2>{_NUMBER})[\s-]*"
    rf"(?:(?P<post_word>{_WORD_CURRENCY})(?![A-Za-z])|(?P<post_sym>{_SYMBOL_CURRENCY}))",
    re.IGNORECASE,
)

_CLAUSE_BREAK = re.compile(r"[.!?](?=\s|$)|[\n;•]")
_QUALIFIERS = frozenset(
    {"under", "over", "below", "above", "up", "less", "more", "than", "budget", "from"}
)
_CENT = Decimal("0.01")


@dataclass(frozen=True)
class PriceMention:
    start: int
    end: int
    amount: Decimal
    currency: str

    @property
    def figure(self) -> str:
        return f"{self.amount} {self.currency}"


@dataclass(frozen=True)
class CatalogProduct:
    id: int
    name: str
    price: Decimal
    currency: str
    pattern: "re.Pattern[str]"


def _normalize_amount(raw: str) -> Optional[Decimal]:
    try:
        return Decimal(raw.replace(",", "")).quantize(_CENT)
    except (InvalidOperation, ValueError):
        return None


def extract_price_mentions(text: str) -> List[PriceMention]:
    """Every currency figure in `text`, with its span, amount and ISO code."""
    mentions: List[PriceMention] = []
    for match in _PRICE_PATTERN.finditer(text):
        number = match.group("n1") or match.group("n2")
        token = (
            match.group("pre_word")
            or match.group("pre_sym")
            or match.group("post_word")
            or match.group("post_sym")
        )
        amount = _normalize_amount(number)
        currency = _CURRENCY_ALIASES.get(token.lower())
        if amount is None or currency is None:
            continue
        mentions.append(PriceMention(match.start(), match.end(), amount, currency))
    return mentions


def extract_price_figures(text: str) -> Set[str]:
    """Normalised figures, e.g. {"4699.00 AED"}."""
    return {m.figure for m in extract_price_mentions(text)}


def build_catalog(products: Iterable[Dict[str, Any]]) -> List[CatalogProduct]:
    """Turn products.json entries into matchable catalog products."""
    catalog: List[CatalogProduct] = []
    for product in products:
        try:
            price = Decimal(str(product["price"])).quantize(_CENT)
            names = [str(product["name"])] + [str(a) for a in product.get("aliases", [])]
        except (KeyError, InvalidOperation, TypeError, ValueError):
            continue
        names = sorted({n.strip() for n in names if n.strip()}, key=len, reverse=True)
        pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(n) for n in names) + r")(?!\w)",
            re.IGNORECASE,
        )
        catalog.append(
            CatalogProduct(
                id=int(product["id"]),
                name=str(product["name"]),
                price=price,
                currency=str(product.get("currency", "AED")).upper(),
                pattern=pattern,
            )
        )
    return catalog


def _clause_breaks(text: str, prices: Sequence[PriceMention]) -> List[int]:
    breaks = []
    for match in _CLAUSE_BREAK.finditer(text):
        pos = match.start()
        if any(p.start <= pos < p.end for p in prices):
            continue  # a period inside "Rs. 4,699" or "4,699.50" is not a sentence end
        breaks.append(pos)
    return breaks


def find_product_price_mismatches(
    reply: str, catalog: Sequence[CatalogProduct]
) -> List[Tuple[int, str]]:
    """
    (product_id, figure) for each figure bound to a product whose catalog
    price/currency it does not match.
    """
    prices = extract_price_mentions(reply)
    if not prices or not catalog:
        return []
    breaks = _clause_breaks(reply, prices)

    def clause_of(pos: int) -> int:
        return bisect.bisect_right(breaks, pos)

    products_found: List[Tuple[int, CatalogProduct]] = []  # (end offset, product)
    for product in catalog:
        for match in product.pattern.finditer(reply):
            products_found.append((match.end(), product))

    mismatches: List[Tuple[int, str]] = []
    for price in prices:
        clause = clause_of(price.start)
        candidates = [
            (end, product)
            for end, product in products_found
            if end <= price.start and clause_of(end - 1) == clause
        ]
        if not candidates:
            continue
        clause_start = breaks[clause - 1] + 1 if clause > 0 else 0
        preceding_words = re.findall(r"[A-Za-z]+", reply[clause_start:price.start])[-3:]
        if any(word.lower() in _QUALIFIERS for word in preceding_words):
            continue
        _, product = max(candidates, key=lambda item: item[0])
        if (price.amount, price.currency) != (product.price, product.currency):
            mismatches.append((product.id, price.figure))
    return mismatches


@dataclass(frozen=True)
class GuardResult:
    reply: str
    ungrounded: Set[str]
    mismatched: List[Tuple[int, str]]

    @property
    def tripped(self) -> bool:
        return bool(self.ungrounded or self.mismatched)


def validate_reply_prices(
    reply: str,
    grounded_texts: Iterable[str],
    user_message: str,
    history: str,
    catalog: Sequence[CatalogProduct],
) -> GuardResult:
    """
    Run both checks. On any failure the reply is replaced by GUARD_REPLY.

    Args:
        reply: The raw LLM reply.
        grounded_texts: Every context block injected this turn.
        user_message: The customer's message (they may quote a budget).
        history: Formatted prior turns.
        catalog: Products for the per-product check.
    """
    allowed: Set[str] = set()
    for text in grounded_texts:
        if text:
            allowed |= extract_price_figures(text)
    allowed |= extract_price_figures(user_message)
    allowed |= extract_price_figures(history)

    ungrounded = extract_price_figures(reply) - allowed
    mismatched = find_product_price_mismatches(reply, catalog)
    if ungrounded or mismatched:
        return GuardResult(GUARD_REPLY, ungrounded, mismatched)
    return GuardResult(reply, set(), [])
