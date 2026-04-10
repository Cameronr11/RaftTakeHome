"""
tests/test_anomaly.py

Tests for the deterministic components of ml/anomaly_detector.py.

We do NOT test the Isolation Forest's predictive performance here —
that belongs in the offline training evaluation script (which already
produces precision/recall/F1 on a held-out test set). Testing model
accuracy in pytest would make tests non-deterministic and dependent
on random seeds across scikit-learn versions.

What we DO test:
  - categorize_items(): pure Python keyword matching — its category
    priority order is invisible in the code and easy to break silently
  - extract_features(): the output shape and one-hot encoding invariants
    that the model depends on at inference time

If categorize_items() silently changes the category for "laptop" from
'tech_equipment' to something else, the trained model receives feature
vectors it was never trained on. These tests catch that regression.
"""

import pytest
import numpy as np

from models.schemas import Order
from ml.anomaly_detector import categorize_items, extract_features, CATEGORIES


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_order(total: float, items: list[str]) -> Order:
    return Order(
        orderId="1001",
        buyer="Test User",
        city="Columbus",
        state="OH",
        total=total,
        items=items,
        raw=None,
    )


# ── categorize_items ───────────────────────────────────────────────────────────

class TestCategorizeItems:

    @pytest.mark.parametrize("items,expected_category", [
        (["laptop"],       "tech_equipment"),
        (["mouse"],        "peripherals"),
        (["router"],       "networking"),
        (["desk lamp"],    "office_supplies"),
        (["tablet"],       "mobile_devices"),
        (["projector"],    "audio_visual"),
        (["coffee maker"], "other"),
        ([],               "other"),
    ])
    def test_category_mapping(self, items, expected_category):
        """
        Each item keyword must resolve to its declared category.
        Parametrized so adding a new category requires adding a row here,
        not a new test function. A single failing row identifies exactly
        which keyword-to-category mapping broke.
        """
        assert categorize_items(items) == expected_category

    def test_first_matching_category_wins(self):
        """
        CATEGORY PRIORITY: when items span multiple categories, the first
        category in CATEGORY_KEYWORDS insertion order wins.
        'gaming pc' hits tech_equipment before 'mouse' hits peripherals.
        If CATEGORY_KEYWORDS is reordered this test catches the silent change.
        """
        assert categorize_items(["gaming pc", "mouse"]) == "tech_equipment"

    def test_keyword_match_is_case_insensitive(self):
        """Items from an LLM might have any casing."""
        assert categorize_items(["LAPTOP"]) == "tech_equipment"

    def test_multi_word_item_matches_contained_keyword(self):
        """'hdmi cable' contains 'hdmi'; 'gaming pc' contains 'gaming'.
        Substring matching must work across word boundaries within an item."""
        assert categorize_items(["hdmi cable"]) == "peripherals"
        assert categorize_items(["gaming pc"])  == "tech_equipment"


# ── extract_features ──────────────────────────────────────────────────────────

class TestExtractFeatures:

    def test_single_order_produces_correct_shape(self):
        """
        Feature matrix must have exactly 1 row and (2 + len(CATEGORIES)) columns.
        The 2 base features are 'total' and 'num_items'.
        If a new category is added to CATEGORIES, this test catches the
        shape change that would make the saved model incompatible.
        """
        orders = [make_order(742.10, ["laptop", "hdmi cable"])]
        X, feature_names = extract_features(orders)
        expected_cols = 2 + len(CATEGORIES)  # total, num_items, + one-hot
        assert X.shape == (1, expected_cols)

    def test_one_hot_is_mutually_exclusive(self):
        """
        Exactly one category column must be 1 per row. Multiple 1s would
        corrupt the feature space the Isolation Forest was trained on.
        """
        orders = [make_order(100.0, ["laptop"])]
        X, feature_names = extract_features(orders)
        cat_cols = [i for i, name in enumerate(feature_names) if name.startswith("cat_")]
        cat_values = X[0, cat_cols]
        assert cat_values.sum() == 1.0

    def test_correct_category_column_is_set(self):
        """For a laptop order, cat_tech_equipment must be 1, all others 0."""
        orders = [make_order(742.10, ["laptop"])]
        X, feature_names = extract_features(orders)
        tech_col = feature_names.index("cat_tech_equipment")
        assert X[0, tech_col] == 1.0

    def test_other_category_column_is_set_for_unknown_items(self):
        orders = [make_order(89.50, ["coffee maker"])]
        X, feature_names = extract_features(orders)
        other_col = feature_names.index("cat_other")
        assert X[0, other_col] == 1.0

    def test_num_items_feature_is_correct(self):
        orders = [make_order(100.0, ["laptop", "hdmi cable", "mouse"])]
        X, feature_names = extract_features(orders)
        num_items_col = feature_names.index("num_items")
        assert X[0, num_items_col] == 3.0

    def test_all_values_are_numeric(self):
        """Feature matrix must contain only finite floats — no NaN or Inf."""
        orders = [make_order(742.10, ["laptop", "hdmi cable"])]
        X, _ = extract_features(orders)
        assert np.all(np.isfinite(X))
