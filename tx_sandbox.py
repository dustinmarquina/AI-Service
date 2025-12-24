# -*- coding: utf-8 -*-
"""
Transaction classifier SANDBOX (edit me!)
- Amount parsing (VN texting)
- Direction (debit/credit)
- Rules-first category
- Optional: Embedding fallback (plug in later)

Run:
  python tx_sandbox.py

What to edit first:
  1) RULES               (add/remove keywords)
  2) CATEGORY_CATALOG    (IDs & names)
  3) THRESHOLD, MARGIN   (decision tuning)
  4) seed_examples()     (when you enable embeddings)
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
import unicodedata  

# -----------------------------
# 0) Category catalog (YOUR IDs)
# -----------------------------
CATEGORY_CATALOG = [
  (1, "Grocery"),
  (2, "Food & Drinks"),
  (3, "Transport"),
  (4, "Fuel"),
  (5, "Utilities"),
  (6, "Telecom"),
  (7, "Rent"),
  (8, "Entertainment"),
  (9, "Healthcare"),
  (10, "Shopping"),
  (11, "Investments"),
  (12, "Fees"),
  (13, "Cash"),
  (14, "Transfer"),
  (15, "Refund"),
  (16, "Other"),
]
CAT_ID = {name: cid for cid, name in CATEGORY_CATALOG}

# -----------------------------
# 1) Amount parsing (VN texting)
# -----------------------------
AMOUNT_PATTERNS = [
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>k|ng[aà]n|ngh[iì]n|nghin|tr|tri[eê]u|m)\b",
    r"(?P<num>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\s*(?P<cur>vnd|vnđ|đ|d)\b",
    r"\b(?P<cur>vnd|vnđ|đ|d)\s*(?P<num>\d{1,3}(?:[.,]\d{3})+|\d+(?:[.,]\d+)?)\b",
    r"(?P<num>\d{1,3}(?:[.,]\d{3})+)\b", # thousands with separators
    r"\b(?P<num>\d+(?:[.,]\d{1,2}))\b", # decimal amounts
    r"\b(?P<num>\d{1,7})\b",  # last resort (guard with context in production)
]

CLEAN_PATTERNS = AMOUNT_PATTERNS[:]  # exclude unit-only patterns
CLEAN_PATTERNS += [
    r"(?P<num>\d+(?:[.,]\d+)?)?\s*(?P<unit>kg|g|ml|l|chai|ph[aầ]n|h[ộo]p|lon|b[ìi]nh)\b",
    r"^[+-]",
    r"\b\d+\b",
]

def normalize_category(name: str) -> str:
    name = name.lower().strip()
    name = name.replace("&", "and")
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"\s*([,.;:])\s*", r"\1 ", name)
    return name.strip()

def remove_accents(input_str: str) -> str:
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])    

def clean_example_text(text: str) -> str:
    """Optional: clean text before embedding (amounts removal, etc.)"""
    # Lowercase
    t = text.lower()
    for pat in CLEAN_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _to_vnd(num_str: str, unit: Optional[str]) -> Optional[int]:
    s = num_str.replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return None
    mult = 1.0
    if unit:
        u = unit.lower()
        if u in {"k","ngan","ngàn","nghìn","nghin"}: mult = 1_000
        elif u in {"tr","trieu","triệu","m"}:       mult = 1_000_000
    return int(round(val * mult))

def extract_amount(text: str) -> Optional[int]:
    t = text.lower()
    for pat in AMOUNT_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if not m: continue
        gd = m.groupdict()
        num = gd.get("num")
        unit = gd.get("unit")
        if num:
            amt = _to_vnd(num, unit)
            if amt is not None and amt >= 0:
                return amt
    return None

# -----------------------------
# 2) Direction (debit/credit)
# -----------------------------
REFUND_RX = re.compile(r"(refund|ho[aà]n\s?ti[eê]n|reversal|chargeback)", re.I)

def parse_direction(text: str) -> str:
    s = text.strip()
    if s.startswith("+"): return "credit"
    if s.startswith("-"): return "debit"
    if REFUND_RX.search(s): return "credit"
    return "debit"

# -----------------------------
# 3) Rules-first classifier
#    (EDIT THESE REGEXES FREELY)
# -----------------------------
RULES: List[Tuple[str, str]] = [
  ("Refund",   r"(refund|ho[aà]n\s?ti[eê]n|reversal|chargeback)"),
  ("Fees",     r"(\bph[ií]\b|maintenance\s?fee|bank\s?charge|ph[ií]\s?duy\s?tr[iì]|ph[ií]\s?th[uươ]̀ng\s?ni[eê]n)"),
  ("Cash",     r"(atm|r[uú]t\s?ti[eê]n|cash\s?adv)"),
  ("Transfer", r"(chuy[eê]n\s?kho[aả]n|transfer|top\s?up|n[aă]p\s?v[ií]|momo|zalopay)"),
  ("Fuel",     r"(x[aă]ng|gas|petrol|tr[aạ]m\s*x[aă]ng)"),
  ("Transport",r"(grab|be\s?car|taxi|bus|xe\s?bu[ýt]|metro|v[eé]\s?xe|uber)"),
  ("Telecom",  r"(viettel|vnpt|fpt\s*telecom|wifi|internet|data|3g|4g|5g|c[ươ]ớc)"),
  ("Utilities",r"(evn|ti[eê]n\s?đi[eê]n|ti[eê]n\s?n[uư][óo]c|n[uư][óo]c)"),
  ("Investments", r"(ssi|tcbs|hsc|binance|btc|ck|ch[ưư]́ng\s?kho[aả]n|coin|usdt)"),
  ("Shopping", r"(shopee|lazada|tiki|mall|qu[aầ]n\s?ao|gi[aà]y|shop|zara|uniqlo|h&m)"),
  ("Food & Drinks", r"(c[oơ]m|x[ôo]i|ph[ơo]|b[áa]nh|m[iì]|g[aà]|cafe|c[àa]\s?ph[êe]|milk\s?tea|tr[àa]\s?s[ữu]a|pizza|kfc|highland|lotteria|grabfood|delivery|tra\s?sua)"),
]
RULES = [(name, re.compile(rx, re.I)) for name, rx in RULES]

def classify_by_rules(text: str) -> Optional[Tuple[int, str]]:
    for name, rx in RULES:
        if rx.search(text):
            return CAT_ID[name], name
    return None

# -----------------------------
# 4) Embedding fallback (OFF by default)
#    Fill these stubs to enable.
# -----------------------------
THRESHOLD = 0.60  # try 0.55 ~ 0.65
MARGIN    = 0.05  # gap to 2nd best

# def seed_examples() -> Dict[str, List[str]]:
#     """Edit: 5–20 short phrases per category (without amounts)."""
#     return {
#       "Food & Drinks": ["com ga","xoi dau","bun bo","pho bo","cafe sua da","tra sua","pizza","kfc"],
#       "Transport": ["grab di truong","taxi","xe buyt","be car","ve xe bus"],
#       "Fuel": ["do xang","tram xang","petrol"],
#       "Utilities": ["tien dien evn","tien nuoc","rac ve sinh"],
#       "Telecom": ["viettel wifi","vnpt internet","fpt telecom","goi 4g"],
#       "Shopping": ["mua ao","mua giay","shopee","lazada","tiki","mall"],
#       "Fees": ["phi duy tri the","bank charge","phi thuong nien"],
#       "Cash": ["rut tien atm","cash advance"],
#       "Transfer": ["chuyen khoan","nap vi momo","top up zalopay"],
#       "Refund": ["hoan tien tiki","refund shopee"],
#       # Others will be anchored by label descriptions if you add that idea later
#     }







    # C) Fallback
    # return {
    #     "raw": raw, "amount": amount, "currency": "VND", "direction": direction,
    #     "categoryId": CAT_ID["Other"], "category_name": "Other/Review",
    #     "confidence": 0.0, "decision_source": "HEURISTIC"
    # }

# -----------------------------
# 6) Demo (edit freely)
# -----------------------------
if __name__ == "__main__":
    samples = [
        # "com ga 12k",
        # "12k com ga",
        # "refund tiki 50k",
        # "rut tien atm 2tr",
        # "viettel wifi 230k",
        # "do xang 80k",
        # "grab di truong 25k",
        # "evn dien 1.2tr",
        # "shopee 120.000",
        # "+50k hoan tien",
        # "-120k phi duy tri the",
        # "beer 45k",
        # "di voi ban 70k"
        "2 phần cơm gà 50k",
    ]
    # for s in samples:
    #     print(s, "→", modelize(s, use_embeddings=True))