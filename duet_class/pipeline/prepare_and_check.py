from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import numpy as np
import pandas as pd

from pandas.tseries.frequencies import to_offset


def _detect_unix_unit(x) -> str:
    """Определение единицы UNIX-времени: 's'|'ms'|'us'|'ns'."""
    try:
        xv = float(x)
    except Exception:
        return 's'
    ax = abs(xv)
    if ax < 1e11:
        return 's'
    if ax < 1e14:
        return 'ms'
    if ax < 1e17:
        return 'us'
    return 'ns'


def _robust_infer_freq(ts: pd.DatetimeIndex) -> Optional[str]:
    """Сначала pd.infer_freq, затем медианный лаг → ближайший Offset."""
    if ts.size < 3:
        return None
    try:
        f = pd.infer_freq(ts)
        if f is not None:
            return f
    except Exception:
        pass
    diffs = np.diff(ts.asi8)
    if diffs.size == 0:
        return None
    med = int(np.median(diffs))
    if med <= 0:
        return None
    try:
        return to_offset(pd.Timedelta(med, unit='ns')).freqstr
    except Exception:
        return None

def _infer_freq_gcd(idx: pd.DatetimeIndex) -> str | None:
    """
    Возвращает частоту как gcd всех положительных лагов.
    Работает устойчиво на рядах типа 1m/2m -> вернёт 1m.
    Если лаги не подходят (нет данных/нулевые/NaT) -> None.
    """
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 3:
        return None

    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return None

    # в секундах (целые)
    sec = (diffs.dt.total_seconds().round().astype("int64")).values
    sec = sec[sec > 0]
    if sec.size == 0:
        return None

    g = int(np.gcd.reduce(sec))
    if g <= 0:
        return None

    try:
        # pandas offset из секунд
        off = pd.to_timedelta(g, unit="s")
        return pd.tseries.frequencies.to_offset(off).freqstr
    except Exception:
        return None


def _norm_colname(x: str) -> str:
    return (
        str(x)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace("/", "")
    )


def _fmt_opt(x: Optional[str]) -> str:
    return x if x is not None and str(x).strip() != "" else "—"


def _fmt_kv_map(m: Dict[str, str], max_items: int = 20) -> str:
    if not m:
        return "—"
    items = list(m.items())
    shown = items[:max_items]
    s = "\n".join([f"    {old} → {new}" for old, new in shown])
    if len(items) > max_items:
        s += f"\n    … ещё {len(items) - max_items}"
    return s


def _fmt_list(xs: List[str], max_items: int = 30) -> str:
    if not xs:
        return "—"
    shown = xs[:max_items]
    s = ", ".join(map(str, shown))
    if len(xs) > max_items:
        s += f", … ещё {len(xs) - max_items}"
    return s


@dataclass
class TimeSeriesPrepReport:
    # Частота и индекс
    inferred_freq: Optional[str] = None
    used_freq: Optional[str] = None
    dropped_duplicates: int = 0
    gap_rows: int = 0

    # ffill метка
    nan_filled_rows: int = 0

    # прогрев (rolling warmup)
    warmup_dropped_rows: int = 0

    # OHLCV
    renamed_columns: Dict[str, str] = field(default_factory=dict)
    missing_ohlcv: List[str] = field(default_factory=list)

    # Свечные проверки
    candle_checks_applied: bool = False
    candle_error_rows: int = 0
    candle_checks_skipped_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        lines: List[str] = []

        lines.append("ОТЧЁТ О ПОДГОТОВКЕ ВРЕМЕННОГО РЯДА")
        lines.append("=" * 44)
        lines.append("")

        # --- Что было сделано ---
        lines.append("Выполненные операции:")
        lines.append("  • Сброс исходного индекса и построение DatetimeIndex")
        lines.append("  • Нормализация временной зоны")
        lines.append("  • Удаление дубликатов по времени")
        if self.used_freq is not None:
            lines.append("  • Восстановление регулярной частоты")
            lines.append("  • Реиндексация и маркировка пропусков (gaps)")
        else:
            lines.append("  • Попытка восстановления частоты (безуспешно)")
        lines.append("  • Заполнение пропусков по правилам (ffill / 0 / target не тронут)")
        lines.append("  • Пометка строк, где ffill заменил NaN (nan_filled)")
        if self.warmup_dropped_rows > 0:
            lines.append("  • Отбрасывание прогрева rolling-фич (warmup)")
        if self.candle_checks_applied:
            lines.append("  • Проверка логической корректности свечей (OHLC)")
        else:
            lines.append("  • Проверка свечей пропущена")
        lines.append("")

        # --- Частота ---
        lines.append("Частота временного ряда:")
        lines.append(f"  Определена автоматически: {_fmt_opt(self.inferred_freq)}")
        lines.append(f"  Использована фактически:  {_fmt_opt(self.used_freq)}")
        lines.append("")

        # --- Индекс и пропуски ---
        lines.append("Качество временного индекса:")
        lines.append(f"  Удалено дубликатов: {int(self.dropped_duplicates)}")
        lines.append(f"  Помечено строк-пропусков (gaps): {int(self.gap_rows)}")
        lines.append(f"  Строк с заполнением NaN через ffill (nan_filled): {int(self.nan_filled_rows)}")
        if self.warmup_dropped_rows > 0:
            lines.append(f"  Отброшено строк прогрева (warmup): {int(self.warmup_dropped_rows)}")
        lines.append("")

        # --- OHLCV ---
        lines.append("Колонки OHLCV:")
        lines.append(f"  Переименованные колонки: {_fmt_kv_map(self.renamed_columns)}")
        lines.append(f"  Отсутствующие обязательные колонки: {_fmt_list(self.missing_ohlcv)}")
        lines.append("")

        # --- Свечные проверки ---
        lines.append("Проверка корректности свечей:")
        if self.candle_checks_applied:
            lines.append("  Проверки выполнены:")
            lines.append("    • High ≥ max(Open, Close)")
            lines.append("    • Low ≤ min(Open, Close)")
            lines.append("    • High ≥ Low")
            lines.append(f"  Обнаружено некорректных свечей: {int(self.candle_error_rows)}")
        else:
            lines.append("  Проверки не выполнялись")
            lines.append(f"  Причина: {self.candle_checks_skipped_reason or '—'}")
        lines.append("")

        return "\n".join(lines)


@dataclass
class FinancialTimeSeriesPreparer:
    tz: str = "UTC"
    timestamp_col: str = "timestamp"
    infer_mode: str = "robust"   # "robust" | "strict"

    drop_warmup: bool = True
    # Если задано, используем эти колонки для определения прогрева; иначе берём все "ffill"-колонки
    warmup_columns: Optional[List[str]] = None

    volume_columns: Optional[List[str]] = field(default_factory=lambda: ["Volume"])
    target_columns: Optional[List[str]] = None

    dedup_agg: Optional[Dict[str, str]] = None
    fallback_freq: Optional[str] = None

    ohlcv_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "Open":   ["open", "o", "opn", "openprice", "priceopen", "openingprice"],
        "High":   ["high", "h", "max", "highprice", "pricehigh"],
        "Low":    ["low", "l", "min", "lowprice", "pricelow"],
        "Close":  ["close", "c", "cls", "closeprice", "priceclose", "closingprice",
                   "adjclose", "adjustedclose", "adj_close", "adj close"],
        "Volume": ["volume", "vol", "v", "basevolume", "quotevolume", "volumeto",
                   "volumefrom", "tradedvolume", "volume_traded"],
    })

    # хранит отчёт последнего prepare()
    _last_report: Optional[TimeSeriesPrepReport] = field(default=None, init=False, repr=False)

    def get_report(self) -> Optional[TimeSeriesPrepReport]:
        """Возвращает отчёт последнего вызова prepare()."""
        return self._last_report

    def _resolve_and_rename_ohlcv(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
        required = ["Open", "High", "Low", "Close", "Volume"]
        present = set(df.columns)

        missing = [c for c in required if c not in present]
        if not missing:
            return df, {}, []

        norm_to_actual: Dict[str, str] = {}
        for c in df.columns:
            norm_to_actual[_norm_colname(c)] = c

        rename_map: Dict[str, str] = {}
        missing_final: List[str] = []

        for std_name in required:
            if std_name in df.columns:
                continue

            candidates = list(self.ohlcv_aliases.get(std_name, [])) + [std_name]
            found_actual: Optional[str] = None
            for cand in candidates:
                key = _norm_colname(cand)
                if key in norm_to_actual:
                    found_actual = norm_to_actual[key]
                    break

            if found_actual is None:
                missing_final.append(std_name)
            else:
                if found_actual != std_name:
                    rename_map[found_actual] = std_name

        if rename_map:
            df = df.rename(columns=rename_map)

        return df, rename_map, missing_final

    def _ensure_datetime_index(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise ValueError("Пустой DataFrame.")
        if self.timestamp_col not in df.columns:
            raise ValueError(f"Колонка '{self.timestamp_col}' не найдена в DataFrame.")

        df = df.reset_index(drop=True).copy()

        unit = _detect_unix_unit(df[self.timestamp_col].iloc[0])
        is_numeric_ts = pd.api.types.is_numeric_dtype(df[self.timestamp_col])

        if is_numeric_ts:
            dt = pd.to_datetime(df[self.timestamp_col], unit=unit, errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(df[self.timestamp_col], errors="coerce", utc=True)

        if dt.isna().any():
            raise ValueError("Некоторые метки времени не удалось преобразовать в datetime.")

        dt = dt.dt.tz_convert(self.tz)
        df.index = dt
        df.sort_index(inplace=True)
        df.index.name = "datetime"
        return df

    def _deduplicate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        if self.dedup_agg:
            dup_cnt = int(df.index.duplicated().sum())
            if dup_cnt:
                df = df.groupby(level=0).agg(self.dedup_agg).sort_index()
            return df, dup_cnt

        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        return df, before - len(df)

    def _infer_and_reindex(self, df: pd.DataFrame):
        inferred = None
        try:
            inferred = pd.infer_freq(df.index)
        except Exception:
            inferred = None

        if inferred is None and self.infer_mode == "robust":
            inferred = _infer_freq_gcd(df.index)

        used = inferred if inferred is not None else self.fallback_freq
        
        if not used:
            # имя индекса всё равно фиксируем
            df.index.name = "datetime"
            return df, inferred, used, df.index

        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=used, tz=df.index.tz)
        full_idx.name = "datetime"  # <- важно
        df = df.reindex(full_idx)
        df.index.name = "datetime"  # <- важно (на случай поведения pandas)
        return df, inferred, used, full_idx

    def _infer_target_columns(self, df: pd.DataFrame) -> Set[str]:
        if self.target_columns is not None:
            return set(self.target_columns)

        # авто: только если явно не задано
        candidates = {"target", "y", "label", "labels", "y_true", "targetvalue", "target_val"}
        out = set()
        for c in df.columns:
            if _norm_colname(c) in candidates:
                out.add(c)
        return out

    def _fill_missing(self, df: pd.DataFrame, report: TimeSeriesPrepReport) -> pd.DataFrame:
        target_cols: Set[str] = self._infer_target_columns(df)

        nan_filled = pd.Series(0, index=df.index, dtype="int8")

        vol_cols = list(self.volume_columns or [])
        if not vol_cols:
            vol_cols = [c for c in df.columns if ("vol" in str(c).lower()) or ("volume" in str(c).lower())]
        vol_cols = [c for c in vol_cols if c in df.columns]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        price_like = [c for c in numeric_cols if (c not in vol_cols) and (c not in target_cols)]
        if price_like:
            was_nan = df[price_like].isna()
            df[price_like] = df[price_like].ffill()
            filled = (was_nan & df[price_like].notna()).any(axis=1).astype("int8")
            nan_filled = (nan_filled | filled).astype("int8")

        if vol_cols:
            for c in vol_cols:
                df[c] = df[c].fillna(0.0)

        non_numeric = [c for c in df.columns if (c not in numeric_cols) and (c not in target_cols)]
        if non_numeric:
            was_nan = df[non_numeric].isna()
            df[non_numeric] = df[non_numeric].ffill()
            filled = (was_nan & df[non_numeric].notna()).any(axis=1).astype("int8")
            nan_filled = (nan_filled | filled).astype("int8")

        df["nan_filled"] = nan_filled
        report.nan_filled_rows = int(nan_filled.sum())
        return df

    def _add_candle_error(self, df: pd.DataFrame, report: TimeSeriesPrepReport) -> pd.DataFrame:
        """
        candle_error = 1 если:
          High < max(Open, Close) OR Low > min(Open, Close) OR High < Low
        Иначе 0.
        Если OHLC нет — candle_error=0 и report фиксирует причину.
        """
        need = ["Open", "High", "Low", "Close"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            df["candle_error"] = 0
            report.candle_checks_applied = False
            report.candle_error_rows = 0
            report.candle_checks_skipped_reason = f"Нет колонок для проверки OHLC: {missing}"
            return df

        # Приводим к числу (если строки) — неконвертируемое станет NaN
        o = pd.to_numeric(df["Open"], errors="coerce")
        h = pd.to_numeric(df["High"], errors="coerce")
        l = pd.to_numeric(df["Low"], errors="coerce")
        c = pd.to_numeric(df["Close"], errors="coerce")

        # Если NaN — НЕ считаем это свечной ошибкой (это уже data quality отдельно),
        # чтобы не смешивать семантику; при желании можно добавить отдельный флаг.
        valid = ~(o.isna() | h.isna() | l.isna() | c.isna())

        cond1 = h < np.maximum(o, c)
        cond2 = l > np.minimum(o, c)
        cond3 = h < l

        err = valid & (cond1 | cond2 | cond3)

        df["candle_error"] = err.astype("int8")
        report.candle_checks_applied = True
        report.candle_error_rows = int(df["candle_error"].sum())
        report.candle_checks_skipped_reason = None
        return df

    def _slice_window(self, df: pd.DataFrame, from_date: Optional[pd.Timestamp], to_date: Optional[pd.Timestamp]) -> pd.DataFrame:
        if from_date is not None:
            frm = pd.Timestamp(from_date)
            frm = frm.tz_convert(self.tz) if frm.tz is not None else frm.tz_localize(self.tz)
            df = df[df.index >= frm]
        if to_date is not None:
            to = pd.Timestamp(to_date)
            to = to.tz_convert(self.tz) if to.tz is not None else to.tz_localize(self.tz)
            df = df[df.index < to]
        return df

    def _drop_warmup_prefix(self, df: pd.DataFrame, report: TimeSeriesPrepReport) -> pd.DataFrame:
        """
        Отбрасывает начальный 'прогрев' rolling-фич: подряд идущие строки в начале,
        где хотя бы в одной выбранной колонке остаётся NaN.

        Колонки для контроля прогрева:
          - если warmup_columns задано: используем их (существующие в df)
          - иначе: все numeric-колонки, кроме Volume/vol_cols, targets и служебных (gaps, nan_filled, candle_error)

        Важно: удаляем только ПРЕФИКС, не трогаем NaN внутри ряда.
        """
        if not self.drop_warmup or df.empty:
            return df

        # targets (явные или авто)
        target_cols = self._infer_target_columns(df)

        # volume cols (как в fill)
        vol_cols = list(self.volume_columns or [])
        if not vol_cols:
            vol_cols = [c for c in df.columns if ("vol" in str(c).lower()) or ("volume" in str(c).lower())]
        vol_cols = [c for c in vol_cols if c in df.columns]

        service = {"gaps", "nan_filled", "candle_error", self.timestamp_col}

        if self.warmup_columns is not None:
            cols = [c for c in self.warmup_columns if c in df.columns]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols = [c for c in numeric_cols if (c not in vol_cols) and (c not in target_cols) and (c not in service)]

        if not cols:
            return df

        nan_any = df[cols].isna().any(axis=1)
        if not bool(nan_any.iloc[0]):
            return df

        # длина префикса, где nan_any=True подряд с начала
        warmup_len = 0
        for v in nan_any.values:
            if v:
                warmup_len += 1
            else:
                break

        if warmup_len <= 0:
            return df

        report.warmup_dropped_rows = warmup_len
        return df.iloc[warmup_len:]

    def prepare(
        self,
        df: pd.DataFrame,
        from_date: Optional[pd.Timestamp] = None,
        to_date: Optional[pd.Timestamp] = None,
        ensure_ohlcv: bool = True,
    ) -> Tuple[pd.DataFrame, TimeSeriesPrepReport]:
        report = TimeSeriesPrepReport()
        work = df.copy()

        if ensure_ohlcv:
            work, renamed, missing_ohlcv = self._resolve_and_rename_ohlcv(work)
            report.renamed_columns = renamed
            report.missing_ohlcv = missing_ohlcv

        work = self._ensure_datetime_index(work)
        work, dropped = self._deduplicate(work)
        report.dropped_duplicates = dropped

        original_idx = work.index
        work, inferred, used, full_idx = self._infer_and_reindex(work)
        report.inferred_freq = inferred
        report.used_freq = used

        # gaps: 1 только для строк, добавленных reindex
        if full_idx.equals(original_idx):
            gaps = pd.Series(0, index=work.index, dtype="int8")
        else:
            missing_idx = full_idx.difference(original_idx)
            gaps = pd.Series(work.index.isin(missing_idx).astype("int8"), index=work.index, dtype="int8")
        work["gaps"] = gaps
        report.gap_rows = int(gaps.sum())

        # fill после gaps (чтобы gaps не менялся)
        work = self._fill_missing(work, report)

        # свечные проверки (после fill, чтобы не ловить ложные ошибки из-за пропусков)
        work = self._add_candle_error(work, report)

        # срез окна
        work = self._slice_window(work, from_date=from_date, to_date=to_date)
        work = self._drop_warmup_prefix(work, report)
        work.index.name = "datetime"

        # сохранить отчёт последнего прогона
        self._last_report = report
        return work, report

    def describe_df(
        self,
        df: pd.DataFrame,
        name: str = "DataFrame",
        top_missing: int = 25,
        as_dict: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Краткая и безопасная диагностика датафрейма.

        Включает:
          - размер и список колонок
          - типы данных (dtypes)
          - индекс: тип, таймзона, диапазон, дубликаты, монотонность
          - частота: infer_freq и робастная
          - пропуски (NaN): всего и топ по колонкам
          - спецколонки: gaps, candle_error (если есть)
        """
        if df is None:
            raise ValueError("df=None")

        n_rows, n_cols = df.shape

        # --- индекс ---
        idx = df.index
        idx_type = type(idx).__name__
        is_dt_index = isinstance(idx, pd.DatetimeIndex)
        tz = str(idx.tz) if is_dt_index and idx.tz is not None else None

        idx_min = idx.max() if n_rows == 0 else None
        idx_max = idx.min() if n_rows == 0 else None
        if n_rows > 0:
            try:
                idx_min = idx.min()
                idx_max = idx.max()
            except Exception:
                idx_min, idx_max = None, None

        idx_is_monotonic = getattr(idx, "is_monotonic_increasing", None)
        idx_dup_count = int(pd.Index(idx).duplicated().sum()) if n_rows > 0 else 0

        # --- частота ---
        infer_freq = None
        robust_freq = None
        if is_dt_index and n_rows >= 3:
            try:
                infer_freq = pd.infer_freq(idx)
            except Exception:
                infer_freq = None
            try:
                robust_freq = _robust_infer_freq(idx)
            except Exception:
                robust_freq = None

        # --- пропуски ---
        na_by_col = df.isna().sum().sort_values(ascending=False)
        total_na = int(na_by_col.sum())
        cols_with_na = int((na_by_col > 0).sum())
        na_top = na_by_col.head(top_missing)

        # --- спецколонки ---
        def _bin_col_info(col: str) -> Optional[Dict[str, Any]]:
            if col not in df.columns:
                return None
            try:
                s = pd.to_numeric(df[col], errors="coerce").fillna(0)
                cnt = int(s.sum())
            except Exception:
                cnt = int(df[col].fillna(0).astype(float).sum())
            return {
                "rows": cnt,
                "share": (cnt / n_rows) if n_rows else None,
            }

        gaps_info = _bin_col_info("gaps")
        candle_info = _bin_col_info("candle_error")
        nan_filled_info = _bin_col_info("nan_filled")

        # --- dtypes ---
        dtypes = df.dtypes.astype(str).to_dict()

        payload: Dict[str, Any] = {
            "name": name,
            "shape": {"rows": n_rows, "cols": n_cols},
            "columns": list(df.columns),
            "dtypes": dtypes,
            "index": {
                "type": idx_type,
                "is_datetimeindex": is_dt_index,
                "tz": tz,
                "min": idx_min,
                "max": idx_max,
                "is_monotonic_increasing": idx_is_monotonic,
                "duplicate_timestamps": idx_dup_count,
            },
            "frequency": {
                "infer_freq": infer_freq,
                "robust_freq": robust_freq,
            },
            "missing": {
                "total_na": total_na,
                "cols_with_na": cols_with_na,
                "top_by_column": na_top.to_dict(),
            },
            "special": {
                "gaps": gaps_info,
                "nan_filled": nan_filled_info,
                "candle_error": candle_info,
            },
        }

        if as_dict:
            return payload

        # --- текстовый вывод ---
        def pct(x: Optional[float]) -> str:
            return "—" if x is None else f"{x*100:.2f}%"

        lines: List[str] = []
        lines.append(f"ИНФОРМАЦИЯ О ДАТАФРЕЙМЕ: {name}")
        lines.append("=" * len(lines[-1]))
        lines.append("")
        lines.append(f"Размер: {n_rows} строк × {n_cols} колонок")
        lines.append("")
        lines.append("Индекс:")
        lines.append(f"  Тип: {idx_type}")
        if is_dt_index:
            lines.append(f"  Таймзона: {tz or '—'}")
        lines.append(f"  Монотонность (возр.): {idx_is_monotonic}")
        lines.append(f"  Дубликаты по времени: {idx_dup_count}")
        lines.append(f"  Диапазон: {_fmt_opt(str(idx_min))} .. {_fmt_opt(str(idx_max))}")
        lines.append("")
        lines.append("Частота:")
        lines.append(f"  infer_freq:  {_fmt_opt(infer_freq)}")
        lines.append(f"  robust_freq: {_fmt_opt(robust_freq)}")
        lines.append("")
        lines.append("Пропуски (NaN):")
        lines.append(f"  Всего NaN: {total_na}")
        lines.append(f"  Колонок с NaN: {cols_with_na}")
        if len(na_top) > 0:
            lines.append("  Топ по колонкам:")
            for col, cnt in na_top.items():
                if cnt > 0:
                    lines.append(f"    {col}: {int(cnt)}")
        lines.append("")
        lines.append("Специальные колонки:")
        if gaps_info is not None:
            lines.append(f"  gaps: {gaps_info['rows']} ({pct(gaps_info['share'])})")
        else:
            lines.append("  gaps: —")
        if nan_filled_info is not None:
            lines.append(f"  nan_filled: {nan_filled_info['rows']} ({pct(nan_filled_info['share'])})")
        else:
            lines.append("  nan_filled: —")
        if candle_info is not None:
            lines.append(f"  candle_error: {candle_info['rows']} ({pct(candle_info['share'])})")
        else:
            lines.append("  candle_error: —")
        lines.append("")
        lines.append("Типы данных (dtypes):")
        dtype_groups: Dict[str, List[str]] = {}
        for col, dt in dtypes.items():
            dtype_groups.setdefault(dt, []).append(col)
        for dt, cols in sorted(dtype_groups.items(), key=lambda x: (-len(x[1]), x[0])):
            preview = ", ".join(cols[:12])
            suffix = f", … ещё {len(cols)-12}" if len(cols) > 12 else ""
            lines.append(f"  {dt}: {len(cols)} кол. ({preview}{suffix})")

        lines.append("")

        return "\n".join(lines)

