"""Unified hotel ontology (MultiLABSA.docx §2, gap #2).

The 6 top-level aspect categories are the ones the two teachers already share
(``utils/label_maps.py``); the 29 sub-categories refine them for the finer
analysis reported in Table 6. Both teachers and the student must map onto the
SAME inventory so every case in the comparison tables is label-compatible.
"""

from __future__ import annotations

from typing import Dict, List

# Re-export the canonical 6 categories + 3 polarities so every experiment uses
# exactly one source of truth (no drift between baselines and the student).
from utils.label_maps import CATEGORIES, SENTIMENTS  # noqa: F401

# 29 sub-categories grouped under the 6 top categories. This is the working
# taxonomy; finalise counts against data_final before freezing Table 1.
SUBCATEGORIES: Dict[str, List[str]] = {
    "FACILITY": ["ROOM", "BATHROOM", "BED", "BUILDING", "LOCATION", "PARKING", "VIEW"],
    "AMENITY": ["BREAKFAST", "FOOD", "BAR", "POOL", "WIFI", "GYM", "SPA"],
    "EXPERIENCE": ["CLEANLINESS", "COMFORT", "NOISE", "ATMOSPHERE", "VALUE"],
    "SERVICE": ["FRONT_DESK", "STAFF", "CHECK_IN_OUT", "HOUSEKEEPING", "CONCIERGE"],
    "LOYALTY": ["MEMBERSHIP", "REBOOK_INTENT"],
    "BRANDING": ["BRAND_TRUST", "RECOMMENDATION", "REPUTATION"],
}

ALL_SUBCATEGORIES: List[str] = [s for subs in SUBCATEGORIES.values() for s in subs]

# Implicit marker used across the pipeline for a NULL aspect/opinion (§4.7).
NULL_TOKEN = "NULL"


def top_category_of(subcategory: str) -> str:
    """Map a sub-category back to its 6-way parent (fallback: FACILITY)."""
    key = (subcategory or "").strip().upper()
    for top, subs in SUBCATEGORIES.items():
        if key in subs:
            return top
    return "FACILITY"
