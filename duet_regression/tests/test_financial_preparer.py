import re
import numpy as np
import pandas as pd
import pytest
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pipeline.prepare_and_check import FinancialTimeSeriesPreparer


# -----------------------
# Helpers / fixtures
# -----------------------

@pytest.fixture
def preparer_utc():
    return FinancialTimeSeriesPreparer(tz="UTC", timestamp_col="timestamp")


@pytest.fixture
def preparer_jerusalem():
    return FinancialTimeSeriesPreparer(tz="Asia/Jerusalem", timestamp_col="timestamp")


def _df_basic_minutes_with_gap_and_dups():
    """
    timestamp: 00:00, 00:01, 00:01 (dup), 00:03  (missing 00:02)
    columns: open/high/low/close/vol + feature + target
    """
    ts = pd.to_datetime([
        "2024-01-01 00:00:00",
        "2024-01-01 00:01:00",
        "2024-01-01 00:01:00",  # duplicate
        "2024-01-01 00:03:00",  # gap at 00:02
    ], utc=True)

    return pd.DataFrame({
        "timestamp": ts,
        "open":   [10.0, 11.0, 99.0, 13.0],   # dup row has different values (should be dropped/aggregated)
        "high":   [10.5, 11.5, 99.5, 13.5],
        "low":    [9.5,  10.5, 98.5, 12.5],
        "close":  [10.2, 11.2, 99.2, 13.2],
        "vol":    [100,  200,  999,  400],
        "feat":   [1.0,  2.0,  999.0, 4.0],
        "target": [0.0,  np.nan, 1.0, np.nan],
    })


def _assert_datetimeindex(df, tzname: str):
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == tzname


# -----------------------
# Core operations
# -----------------------

def test_always_reset_index_and_build_datetimeindex_from_timestamp(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()

    # Сделаем "грязный" индекс до вызова
    df2 = df.copy()
    df2.index = [100, 101, 102, 103]

    out, report = preparer_utc.prepare(df2)

    _assert_datetimeindex(out, "UTC")
    assert out.index.name == "datetime"
    # timestamp_col может остаться колонкой — это ок (не требовалось удалять)
    assert "timestamp" in out.columns


def test_timezone_normalization(preparer_jerusalem):
    df = _df_basic_minutes_with_gap_and_dups()

    out, report = preparer_jerusalem.prepare(df)

    _assert_datetimeindex(out, "Asia/Jerusalem")
    # Проверяем что время корректно конвертировалось из UTC
    # 2024-01-01 00:00 UTC = 2024-01-01 02:00 Asia/Jerusalem (зимой обычно UTC+2)
    assert out.index.min().hour in (2, 3)  # на случай DST/источника


def test_remove_duplicates_keep_first_default(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)

    assert report.dropped_duplicates == 1
    # Для 00:01 должна остаться первая строка: open=11.0, vol=200
    ts_001 = pd.Timestamp("2024-01-01 00:01:00", tz="UTC")
    assert float(out.loc[ts_001, "Open"]) == 11.0
    assert float(out.loc[ts_001, "Volume"]) == 200.0


def test_deduplicate_with_aggregation():
    df = _df_basic_minutes_with_gap_and_dups()
    prep = FinancialTimeSeriesPreparer(
        tz="UTC",
        timestamp_col="timestamp",
        dedup_agg={"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum", "feat": "last", "target": "last"},
    )
    out, report = prep.prepare(df)

    assert report.dropped_duplicates == 1
    ts_001 = pd.Timestamp("2024-01-01 00:01:00", tz="UTC")
    # Volume должен суммироваться: 200 + 999 = 1199
    assert float(out.loc[ts_001, "Volume"]) == 1199.0
    # High = max(11.5, 99.5) = 99.5
    assert float(out.loc[ts_001, "High"]) == 99.5
    # Close = last => 99.2
    assert float(out.loc[ts_001, "Close"]) == 99.2


def test_restore_regular_frequency_and_gaps_marking(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)

    # Должна появиться строка на 00:02
    ts_002 = pd.Timestamp("2024-01-01 00:02:00", tz="UTC")
    assert ts_002 in out.index

    # gaps: 1 только для добавленной строки
    assert out.loc[ts_002, "gaps"] == 1
    # Остальные оригинальные должны быть 0
    ts_000 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    assert out.loc[ts_000, "gaps"] == 0
    assert report.gap_rows >= 1


def test_fill_rules_prices_ffill_volume_zero_targets_untouched(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)

    ts_002 = pd.Timestamp("2024-01-01 00:02:00", tz="UTC")

    # Price/feature: ffill
    assert float(out.loc[ts_002, "Close"]) == float(out.loc[pd.Timestamp("2024-01-01 00:01:00", tz="UTC"), "Close"])
    assert float(out.loc[ts_002, "feat"]) == float(out.loc[pd.Timestamp("2024-01-01 00:01:00", tz="UTC"), "feat"])

    # Volume: 0
    assert float(out.loc[ts_002, "Volume"]) == 0.0

    # target: не трогать (должен остаться NaN на вставленной строке)
    assert pd.isna(out.loc[ts_002, "target"])


def test_slice_window_half_open_interval(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(
        df,
        from_date="2024-01-01 00:01:00+00:00",
        to_date="2024-01-01 00:03:00+00:00",
    )

    assert out.index.min() == pd.Timestamp("2024-01-01 00:01:00", tz="UTC")
    # to_date исключается => 00:03 не должно быть
    assert pd.Timestamp("2024-01-01 00:03:00", tz="UTC") not in out.index


# -----------------------
# OHLCV alias + missing reporting
# -----------------------

def test_ohlcv_alias_renaming(preparer_utc):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 00:00:00"], utc=True),
        "OPEN_price": [1.0],
        "HIGH": [2.0],
        "low_price": [0.5],
        "Adj Close": [1.5],
        "vol": [10],
    })
    out, report = preparer_utc.prepare(df, ensure_ohlcv=True)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        assert c in out.columns

    assert report.missing_ohlcv == []
    # renamed_columns должен содержать хотя бы какие-то переименования
    assert len(report.renamed_columns) >= 1


def test_missing_ohlcv_is_reported(preparer_utc):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 00:00:00"], utc=True),
        "price": [1.0],
    })
    out, report = preparer_utc.prepare(df, ensure_ohlcv=True)

    # Должны явно сообщиться отсутствующие
    assert set(report.missing_ohlcv) == {"Open", "High", "Low", "Close", "Volume"}


# -----------------------
# Candle checks
# -----------------------

def test_candle_error_detects_inconsistencies(preparer_utc):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 00:01:00"], utc=True),
        "Open":  [10.0, 10.0],
        "Close": [12.0, 12.0],
        "High":  [11.0, 13.0],  # 1я свеча: High=11 < max(10,12)=12 => ошибка
        "Low":   [9.0,  9.0],
        "Volume":[1, 1],
    })

    out, report = preparer_utc.prepare(df, ensure_ohlcv=False)

    assert "candle_error" in out.columns
    assert int(out["candle_error"].sum()) == 1
    assert report.candle_checks_applied is True
    assert report.candle_error_rows == 1


def test_candle_checks_skipped_when_missing_cols(preparer_utc):
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-01-01 00:00:00"], utc=True),
        "Close": [1.0],  # нет Open/High/Low
    })
    out, report = preparer_utc.prepare(df, ensure_ohlcv=False)

    assert "candle_error" in out.columns
    assert int(out["candle_error"].sum()) == 0
    assert report.candle_checks_applied is False
    assert report.candle_checks_skipped_reason is not None
    assert "Нет колонок" in report.candle_checks_skipped_reason


# -----------------------
# Error handling / problematic data
# -----------------------

def test_error_on_empty_df(preparer_utc):
    with pytest.raises(ValueError, match="Пустой DataFrame"):
        preparer_utc.prepare(pd.DataFrame())


def test_error_on_missing_timestamp_col(preparer_utc):
    df = pd.DataFrame({"ts": [1, 2, 3], "Close": [1, 2, 3]})
    with pytest.raises(ValueError, match="Колонка 'timestamp'"):
        preparer_utc.prepare(df, ensure_ohlcv=False)


def test_error_on_unparsable_timestamps(preparer_utc):
    df = pd.DataFrame({
        "timestamp": ["not-a-date", "still-bad"],
        "Close": [1.0, 2.0],
    })
    with pytest.raises(ValueError, match="не удалось преобразовать"):
        preparer_utc.prepare(df, ensure_ohlcv=False)


def test_no_freq_inferred_and_no_fallback_keeps_index_no_gaps(preparer_utc):
    # Нерегулярный ряд: интервалы 1 мин, 7 мин, 2 мин
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 00:01:00",
            "2024-01-01 00:08:00",
            "2024-01-01 00:10:00",
        ], utc=True),
        "Close": [1.0, 2.0, 3.0, 4.0],
        "Volume": [1, 1, 1, 1],
    })

    prep = FinancialTimeSeriesPreparer(
        tz="UTC",
        timestamp_col="timestamp",
        fallback_freq=None,
        infer_mode="strict",
    )
    out, report = prep.prepare(df, ensure_ohlcv=False)
    assert report.used_freq is None
    assert int(out["gaps"].sum()) == 0


def test_fallback_freq_used_when_infer_fails():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-01-01 00:00:00",
            "2024-01-01 00:01:00",
            "2024-01-01 00:08:00",
        ], utc=True),
        "Close": [1.0, 2.0, 3.0],
        "Volume": [1, 1, 1],
    })

    prep = FinancialTimeSeriesPreparer(
        tz="UTC",
        timestamp_col="timestamp",
        fallback_freq="1T",
        infer_mode="strict",
    )
    out, report = prep.prepare(df, ensure_ohlcv=False)
    assert report.used_freq in ("1T", "T", "min", "1min")
    assert int(out["gaps"].sum()) >= 1


# -----------------------
# Report / printing / get_report
# -----------------------

def test_get_report_returns_last_report(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)
    last = preparer_utc.get_report()
    assert last is report


def test_report_print_is_russian_and_mentions_work_done(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)

    txt = str(report)
    # базовые русские маркеры
    assert "ОТЧЁТ" in txt or "Отчёт" in txt
    assert "Выполненные операции" in txt
    assert "Колонки OHLCV" in txt
    assert "Проверка" in txt


# -----------------------
# describe_df (lightweight)
# -----------------------

def test_describe_df_text_contains_key_sections(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)

    s = preparer_utc.describe_df(out, name="После подготовки")
    assert "ИНФОРМАЦИЯ О ДАТАФРЕЙМЕ" in s
    assert "Индекс" in s
    assert "Частота" in s
    assert "Пропуски" in s
    assert "Типы данных" in s
    # убедимся что нет секций nunique/describe (вы просили убрать)
    assert "Уникальность" not in s
    assert "Статистика по числовым" not in s


def test_describe_df_as_dict_has_expected_keys(preparer_utc):
    df = _df_basic_minutes_with_gap_and_dups()
    out, report = preparer_utc.prepare(df)

    d = preparer_utc.describe_df(out, as_dict=True)
    assert "shape" in d
    assert "index" in d
    assert "missing" in d
    assert "special" in d
    assert "dtypes" in d
    assert "frequency" in d
