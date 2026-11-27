import numpy as np

from canteen import cleaning


def test_clean_train_applies_every_rule(raw_df, params):
    later = raw_df.iloc[:20].drop_duplicates()
    rules = cleaning.learn_rules(raw_df, later, later.drop(columns="plates_sold"), params)
    out = cleaning.clean(raw_df, rules, params, is_train=True)
    assert not out.duplicated().any()
    assert (out["exam_days_left"] == 999).sum() == 0 and out["exam_days_left"].isna().sum() == 10
    stuck = (out["date"] >= "2026-01-10") & (out["date"] <= "2026-01-20")
    assert out.loc[stuck, "menu_price_rs"].isna().all()
    assert list(out["date"]) == sorted(out["date"])


def test_clean_other_files_keep_all_rows(raw_df, params):
    rules = cleaning.learn_rules(raw_df, raw_df.iloc[:20], raw_df.iloc[:20].drop(columns="plates_sold"), params)
    hold = raw_df.drop(columns="plates_sold").iloc[:40]
    out = cleaning.clean(hold, rules, params, is_train=False)
    assert len(out) == len(hold.drop_duplicates())
    assert out["menu_price_rs"].notna().all()  # stuck window only applies to train


def test_spelling_variants_are_mapped(params):
    import pandas as pd
    df = pd.DataFrame({"day_type": ["Weekday ", "SATURDAY", "holiday.", "weekday"], "block": ["a-block", "B-Block", "hostel ", "Hostel"]})
    out = cleaning.fix_spelling(df, params["data"]["levels"])
    assert set(out["day_type"]) <= {"weekday", "saturday", "holiday"}
    assert set(out["block"]) <= {"A-Block", "B-Block", "Hostel"}
