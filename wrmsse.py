"""WRMSSE evaluator for M5 Forecasting Accuracy (all 12 aggregation levels)."""

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
HORIZON = 28

LEVELS = {
    "1 total": [], "2 state": ["state_id"], "3 store": ["store_id"],
    "4 category": ["cat_id"], "5 department": ["dept_id"],
    "6 state × category": ["state_id", "cat_id"], "7 state × department": ["state_id", "dept_id"],
    "8 store × category": ["store_id", "cat_id"], "9 store × department": ["store_id", "dept_id"],
    "10 item": ["item_id"], "11 item × state": ["item_id", "state_id"], "12 item × store": ["item_id", "store_id"],
}


class WRMSSE:
    """
    sales_wide: sales_train_evaluation.csv as loaded
    prices:     sell_prices.csv as loaded
    calendar:   calendar.csv with an int `day` column (d_1 -> 1)
    train_end:  last training day; weights use its final 28 days
    """

    def __init__(self, sales_wide, prices, calendar, train_end):
        self.ids = sales_wide[ID_COLS].astype(str).reset_index(drop=True)
        train = sales_wide[[f"d_{d}" for d in range(1, train_end + 1)]].to_numpy(np.float64)

        last_days = range(train_end - HORIZON + 1, train_end + 1)
        weeks = calendar.set_index("day").loc[list(last_days), "wm_yr_wk"].to_numpy()
        price_wide = (
            prices.assign(store_id=prices["store_id"].astype(str), item_id=prices["item_id"].astype(str))
            .pivot(index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price")
            .reindex(pd.MultiIndex.from_frame(self.ids[["store_id", "item_id"]]))[weeks]
        )
        dollars = (train[:, -HORIZON:] * price_wide.fillna(0).to_numpy()).sum(axis=1)

        self.levels = {}
        for name, cols in LEVELS.items():
            key = self.ids.groupby(cols, sort=True).ngroup().to_numpy() if cols else None
            agg = self._agg(train, key)
            diffs = np.diff(agg, axis=1) ** 2
            started = np.cumsum(agg[:, :-1] > 0, axis=1) > 0  # from each series' first sale
            scale = (diffs * started).sum(axis=1) / started.sum(axis=1)
            weight = self._agg(dollars[:, None], key)[:, 0]
            self.levels[name] = (key, scale, weight / weight.sum())

    @staticmethod
    def _agg(values, key):
        if key is None:
            return values.sum(axis=0, keepdims=True)
        return pd.DataFrame(values).groupby(key).sum().to_numpy()

    def score(self, pred, actual):
        """pred, actual: arrays (30,490 × 28) in sales_wide row order. Returns (WRMSSE, per-level Series)."""
        per_level = {}
        for name, (key, scale, weight) in self.levels.items():
            err = ((self._agg(actual, key) - self._agg(pred, key)) ** 2).mean(axis=1)
            per_level[name] = np.nansum(weight * np.sqrt(err / scale))
        per_level = pd.Series(per_level, name="WRMSSE")
        return per_level.mean(), per_level
