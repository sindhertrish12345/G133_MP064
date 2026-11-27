from canteen import data_checks as dc


def test_sentinel_found(raw_df, params):
    res = dc.check_sentinel_value(raw_df, params)
    assert res == {"present": True, "column": "exam_days_left", "value": 999, "count": 10}


def test_duplicates_counted(raw_df, params):
    assert dc.check_duplicate_rows(raw_df) == {"present": True, "count": 3}


def test_stuck_sensor_window(raw_df, params):
    res = dc.check_stuck_sensor(raw_df, params)
    assert res["present"] and res["column"] == "menu_price_rs"
    assert (res["start_date"], res["end_date"]) == ("2026-01-10", "2026-01-20")


def test_clean_data_reports_absent(raw_df, params):
    clean = raw_df.drop_duplicates().copy()
    clean.loc[clean["exam_days_left"] == 999, "exam_days_left"] = 10
    clean["menu_price_rs"] = range(30, 30 + len(clean))
    assert not dc.check_sentinel_value(clean, params)["present"]
    assert not dc.check_duplicate_rows(clean)["present"]
    assert not dc.check_stuck_sensor(clean, params)["present"]
    assert not dc.check_target_typos(clean, params)["present"]


def test_target_typos_detected(raw_df, params):
    df = raw_df.drop_duplicates().copy()
    df.loc[[3, 4], "plates_sold"] *= 10
    assert dc.check_target_typos(df, params) == {"present": True, "count": 2}
