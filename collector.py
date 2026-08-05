#!/usr/bin/env python3
"""Pionex BTC Radar Data Collector v0.2.

Read-only collector: Pionex BTC_USDT Spot PRIMARY, Binance BTCUSDT Spot
fallback, Coinbase/CoinGecko controls. Uses completed 4H/1D candles, computes
EMA50/200, DMI(14)/ADX(6), OBV/MAOBV30 and ATR14, and writes JSON artefacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests

METHOD_ID = "BTCRADAR_OHLCV_V0_2"
METHOD_VERSION = "0.2.0"
TZ_NAME = "Europe/Prague"
PIONEX_SYMBOL = "BTC_USDT"
BINANCE_SYMBOL = "BTCUSDT"
CANDLE_LIMIT = 500
MIN_CANDLES = 200
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
BASELINES = DATA / "baselines"
LATEST = DATA / "latest.json"
RUN_LOG = DATA / "run_log.json"

PIONEX = "https://api.pionex.com"
BINANCE_HOSTS = [
    "https://data-api.binance.vision",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
    "https://api.binance.com",
]

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "pionex-btc-radar/0.2",
    "Accept": "application/json",
    "Cache-Control": "no-cache",
})


class CollectorError(RuntimeError):
    def __init__(self, message: str, error_class: str, stage: str, details: Any = None):
        super().__init__(message)
        self.error_class = error_class
        self.stage = stage
        self.details = details or {}

    def record(self, source: str) -> dict[str, Any]:
        return {
            "source": source,
            "error_class": self.error_class,
            "stage": self.stage,
            "message": str(self),
            "details": self.details,
        }


def get_json(url: str, params: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    attempts = []
    for n in range(1, 3):
        started = time.monotonic()
        try:
            response = SESSION.get(url, params=params, timeout=20)
            attempts.append({
                "attempt": n,
                "status": response.status_code,
                "elapsed_ms": round((time.monotonic() - started) * 1000),
            })
            response.raise_for_status()
            return response.json(), {
                "url": response.url,
                "attempts": attempts,
                "obtained_at": datetime.now(timezone.utc).isoformat(),
            }
        except requests.HTTPError as exc:
            if n == 2:
                raise CollectorError(str(exc), "HTTP_OR_RATE_LIMIT", "HTTP", attempts) from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            attempts.append({"attempt": n, "reason": str(exc)})
            if n == 2:
                raise CollectorError(str(exc), "NETWORK_OR_ACCESS", "HTTP", attempts) from exc
        except (ValueError, requests.RequestException) as exc:
            raise CollectorError(str(exc), "SCHEMA_OR_PARSE", "HTTP", attempts) from exc
        time.sleep(n)
    raise CollectorError("Unexpected request failure", "UNCLASSIFIED_SOURCE_FAILURE", "HTTP")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def sha256(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(blob).hexdigest()


def validate_candles(df: pd.DataFrame, interval_ms: int, label: str, now_ms: int) -> pd.DataFrame:
    required = ["open_time", "close_time", "open", "high", "low", "close", "volume"]
    if any(column not in df.columns for column in required):
        raise CollectorError("Missing OHLCV field", "SCHEMA_OR_PARSE", f"VALIDATE_{label}")
    df = df[required + (["quote_volume"] if "quote_volume" in df.columns else [])].copy()
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["open_time"] = pd.to_numeric(df["open_time"], errors="raise").astype("int64")
    df["close_time"] = pd.to_numeric(df["close_time"], errors="raise").astype("int64")
    df = df[df["close_time"] <= now_ms].drop_duplicates("open_time", keep="last")
    df = df.sort_values("open_time").reset_index(drop=True)
    if len(df) < MIN_CANDLES:
        raise CollectorError(
            f"Only {len(df)} completed {label} candles; {MIN_CANDLES} required",
            "PAGINATION_OR_HISTORY_LENGTH", f"VALIDATE_{label}", {"count": len(df)},
        )
    gaps = df["open_time"].diff().dropna()
    if not gaps.eq(interval_ms).all():
        bad = gaps[~gaps.eq(interval_ms)].head().tolist()
        raise CollectorError("Candle gaps/order mismatch", "CANDLE_GAP_OR_ORDER", f"VALIDATE_{label}", bad)
    if ((df["high"] < df[["open", "close"]].max(axis=1)) | (df["low"] > df[["open", "close"]].min(axis=1))).any():
        raise CollectorError("Invalid OHLC relationship", "SCHEMA_OR_PARSE", f"VALIDATE_{label}")
    return df


def fetch_pionex(now_ms: int) -> dict[str, Any]:
    symbol_errors = []
    symbol = None
    symbol_meta = None
    for params in ({"symbols": PIONEX_SYMBOL, "type": "SPOT"}, {"symbol": PIONEX_SYMBOL, "type": "SPOT"}):
        try:
            payload, symbol_meta = get_json(f"{PIONEX}/api/v1/common/symbols", params)
            symbols = (payload.get("data") or {}).get("symbols") or []
            symbol = next((x for x in symbols if x.get("symbol") == PIONEX_SYMBOL and x.get("type", "SPOT") == "SPOT"), None)
            if payload.get("result") is True and symbol and symbol.get("enable", True):
                break
            symbol = None
        except CollectorError as exc:
            symbol_errors.append(exc.record("PIONEX"))
    if not symbol:
        raise CollectorError("Pionex BTC_USDT Spot identity failed", "SYMBOL_OR_MARKET_IDENTITY", "VERIFY_SYMBOL", symbol_errors)

    tick_payload, tick_meta = get_json(f"{PIONEX}/api/v1/market/tickers", {"symbol": PIONEX_SYMBOL, "type": "SPOT"})
    tickers = (tick_payload.get("data") or {}).get("tickers") or []
    ticker_raw = next((x for x in tickers if x.get("symbol") == PIONEX_SYMBOL), None)
    if tick_payload.get("result") is not True or not ticker_raw:
        raise CollectorError("Pionex ticker missing", "EMPTY_OR_INCOMPLETE_RESPONSE", "FETCH_TICKER")
    open_price, price = float(ticker_raw["open"]), float(ticker_raw["close"])
    ticker = {
        "price": price, "open_24h": open_price,
        "change_24h_pct": (price / open_price - 1) * 100 if open_price else None,
        "high_24h": float(ticker_raw["high"]), "low_24h": float(ticker_raw["low"]),
        "base_volume_24h": float(ticker_raw["volume"]),
        "quote_volume_24h": float(ticker_raw["amount"]),
        "trade_count_24h": int(ticker_raw["count"]) if ticker_raw.get("count") is not None else None,
        "source_time_ms": int(ticker_raw["time"]) if ticker_raw.get("time") else None,
    }

    frames, metas = {}, {}
    for interval, label, step in (("4H", "4h", 14_400_000), ("1D", "1d", 86_400_000)):
        payload, meta = get_json(f"{PIONEX}/api/v1/market/klines", {"symbol": PIONEX_SYMBOL, "interval": interval, "limit": CANDLE_LIMIT})
        rows = (payload.get("data") or {}).get("klines") or []
        if payload.get("result") is not True or not rows:
            raise CollectorError(f"Pionex {interval} klines missing", "EMPTY_OR_INCOMPLETE_RESPONSE", "FETCH_KLINES")
        records = []
        for row in rows:
            open_time = int(row["time"])
            records.append({
                "open_time": open_time, "close_time": open_time + step - 1,
                "open": row["open"], "high": row["high"], "low": row["low"],
                "close": row["close"], "volume": row["volume"],
                "quote_volume": row.get("amount"),
            })
        frames[label] = validate_candles(pd.DataFrame(records), step, label, now_ms)
        metas[label] = meta
    return {
        "venue": "PIONEX", "role": "PRIMARY", "qualification": "TESTING",
        "hostname": "api.pionex.com", "instrument": PIONEX_SYMBOL, "market": "SPOT",
        "ticker": ticker, "4h": frames["4h"], "1d": frames["1d"],
        "meta": {"symbol": symbol_meta, "ticker": tick_meta, **metas},
    }


def fetch_binance_host(host: str, now_ms: int) -> dict[str, Any]:
    exchange, symbol_meta = get_json(f"{host}/api/v3/exchangeInfo", {"symbol": BINANCE_SYMBOL})
    symbol = next((x for x in exchange.get("symbols", []) if x.get("symbol") == BINANCE_SYMBOL and x.get("status") == "TRADING" and x.get("isSpotTradingAllowed", True)), None)
    if not symbol:
        raise CollectorError("Binance BTCUSDT Spot identity failed", "SYMBOL_OR_MARKET_IDENTITY", "VERIFY_SYMBOL")
    raw, tick_meta = get_json(f"{host}/api/v3/ticker/24hr", {"symbol": BINANCE_SYMBOL})
    ticker = {
        "price": float(raw["lastPrice"]), "open_24h": float(raw["openPrice"]),
        "change_24h_pct": float(raw["priceChangePercent"]),
        "high_24h": float(raw["highPrice"]), "low_24h": float(raw["lowPrice"]),
        "base_volume_24h": float(raw["volume"]), "quote_volume_24h": float(raw["quoteVolume"]),
        "trade_count_24h": int(raw["count"]), "source_time_ms": int(raw["closeTime"]),
    }
    frames, metas = {}, {}
    for interval, label, step in (("4h", "4h", 14_400_000), ("1d", "1d", 86_400_000)):
        rows, meta = get_json(f"{host}/api/v3/klines", {"symbol": BINANCE_SYMBOL, "interval": interval, "limit": CANDLE_LIMIT, "timeZone": "0"})
        records = [{
            "open_time": int(x[0]), "open": x[1], "high": x[2], "low": x[3], "close": x[4],
            "volume": x[5], "close_time": int(x[6]), "quote_volume": x[7],
        } for x in rows]
        frames[label] = validate_candles(pd.DataFrame(records), step, label, now_ms)
        metas[label] = meta
    return {
        "venue": "BINANCE", "role": "APPROVED_FALLBACK", "qualification": "TESTING",
        "hostname": host.removeprefix("https://"), "instrument": BINANCE_SYMBOL, "market": "SPOT",
        "ticker": ticker, "4h": frames["4h"], "1d": frames["1d"],
        "meta": {"symbol": symbol_meta, "ticker": tick_meta, **metas},
    }


def fetch_binance(now_ms: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors = []
    for host in BINANCE_HOSTS:
        try:
            return fetch_binance_host(host, now_ms), errors
        except CollectorError as exc:
            record = exc.record("BINANCE")
            record["hostname"] = host.removeprefix("https://")
            errors.append(record)
    raise CollectorError("All Binance transport paths failed", "TECHNICAL_ACQUISITION_FAILURE", "BINANCE_FALLBACK", errors)


def wilder(series: pd.Series, period: int) -> pd.Series:
    values = series.astype(float).tolist()
    out = [math.nan] * len(values)
    valid = [i for i, value in enumerate(values) if not math.isnan(value)]
    if len(valid) < period:
        return pd.Series(out, index=series.index, dtype=float)
    seed_indices = valid[:period]
    previous = sum(values[i] for i in seed_indices) / period
    out[seed_indices[-1]] = previous
    for i in valid[period:]:
        previous = (previous * (period - 1) + values[i]) / period
        out[i] = previous
    return pd.Series(out, index=series.index, dtype=float)


def indicators(df: pd.DataFrame) -> dict[str, Any]:
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    prev_close = close.shift(1)
    tr = pd.concat([(high-low), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    up, down = high.diff(), -low.diff()
    plus_dm = pd.Series(0.0, index=df.index).where(~((up > down) & (up > 0)), up)
    minus_dm = pd.Series(0.0, index=df.index).where(~((down > up) & (down > 0)), down)
    atr = wilder(tr, 14)
    plus_rma, minus_rma = wilder(plus_dm, 14), wilder(minus_dm, 14)
    pdi, mdi = 100 * plus_rma / atr, 100 * minus_rma / atr
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, math.nan)
    adx = wilder(dx, 6)
    obv = [0.0]
    for i in range(1, len(df)):
        obv.append(obv[-1] + volume.iloc[i] if close.iloc[i] > close.iloc[i-1] else obv[-1] - volume.iloc[i] if close.iloc[i] < close.iloc[i-1] else obv[-1])
    obv_s = pd.Series(obv, index=df.index)
    maobv = obv_s.rolling(30).mean()
    last = lambda s: float(s.dropna().iloc[-1])
    result = {
        "candle_count": int(len(df)),
        "last_complete_candle_end": datetime.fromtimestamp(int(df.iloc[-1]["close_time"]) / 1000, timezone.utc).isoformat(),
        "close": float(close.iloc[-1]), "ema50": last(ema50), "ema200": last(ema200),
        "pdi": last(pdi), "mdi": last(mdi), "adx": last(adx),
        "obv": float(obv_s.iloc[-1]), "maobv30": last(maobv), "atr14": last(atr),
    }
    result["ema_spread_pct"] = (result["ema50"] / result["ema200"] - 1) * 100
    result["atr_pct"] = result["atr14"] / result["close"] * 100
    result["obv_relation"] = "ABOVE" if result["obv"] > result["maobv30"] else "BELOW" if result["obv"] < result["maobv30"] else "EQUAL"
    delta = obv_s.iloc[-1] - obv_s.iloc[-6]
    result["obv_direction"] = "UP" if delta > 0 else "DOWN" if delta < 0 else "FLAT"
    ema_sign = 1 if result["close"] > result["ema50"] > result["ema200"] else -1 if result["close"] < result["ema50"] < result["ema200"] else 0
    dmi_sign = 1 if result["pdi"] > result["mdi"] and result["adx"] >= 20 else -1 if result["mdi"] > result["pdi"] and result["adx"] >= 20 else 0
    obv_sign = 1 if result["obv_relation"] == "ABOVE" and result["obv_direction"] == "UP" else -1 if result["obv_relation"] == "BELOW" and result["obv_direction"] == "DOWN" else 0
    score = ema_sign + dmi_sign + obv_sign
    result["evidence"] = {"ema_sign": ema_sign, "dmi_sign": dmi_sign, "obv_sign": obv_sign, "score": score, "timeframe_direction": "BULL" if score >= 2 else "BEAR" if score <= -2 else "SIDEWAYS"}
    return result


def classify(h4: dict[str, Any], d1: dict[str, Any]) -> dict[str, Any]:
    h4_score = int(h4["evidence"]["score"])
    d1_score = int(d1["evidence"]["score"])
    score = h4_score + d1_score
    direction = "BULL" if score >= 2 else "BEAR" if score <= -2 else "SIDEWAYS"
    sign = 1 if direction == "BULL" else -1 if direction == "BEAR" else 0

    tf4 = h4["evidence"]["timeframe_direction"]
    tf1 = d1["evidence"]["timeframe_direction"]
    bias4 = "BULL" if h4_score > 0 else "BEAR" if h4_score < 0 else "NEUTRAL"
    bias1 = "BULL" if d1_score > 0 else "BEAR" if d1_score < 0 else "NEUTRAL"

    timeframe_tension = bias4 in {"BULL", "BEAR"} and bias1 in {"BULL", "BEAR"} and bias4 != bias1
    timeframe_alignment = bias4 in {"BULL", "BEAR"} and bias4 == bias1
    divergence = tf4 in {"BULL", "BEAR"} and tf1 in {"BULL", "BEAR"} and tf4 != tf1

    character = (
        "CHAOS"
        if divergence
        else "TREND"
        if direction != "SIDEWAYS" and h4["adx"] >= 25 and d1["adx"] >= 20
        else "CONSOLIDATION"
    )
    checks = {
        "ema": sign != 0 and h4["evidence"]["ema_sign"] == sign == d1["evidence"]["ema_sign"],
        "dmi_adx": sign != 0 and h4["evidence"]["dmi_sign"] == sign == d1["evidence"]["dmi_sign"],
        "obv": sign != 0 and h4["evidence"]["obv_sign"] == sign == d1["evidence"]["obv_sign"],
        "timeframe_alignment": timeframe_alignment,
    }
    count = sum(checks.values())
    strength = "WEAK" if count <= 1 else "MEDIUM" if count == 2 else "STRONG"
    if (divergence or timeframe_tension) and strength == "STRONG":
        strength = "MEDIUM"

    return {
        "direction": direction,
        "character": character,
        "signal_strength": strength,
        "confirmation_count": count,
        "confirmations": checks,
        "timeframe_direction_4h": tf4,
        "timeframe_direction_1d": tf1,
        "timeframe_bias_4h": bias4,
        "timeframe_bias_1d": bias1,
        "timeframe_alignment": timeframe_alignment,
        "timeframe_tension": timeframe_tension,
        "timeframe_divergence": divergence,
        "raw_direction_score": score,
        "raw_direction_score_4h": h4_score,
        "raw_direction_score_1d": d1_score,
        "method_id": METHOD_ID,
        "method_version": METHOD_VERSION,
    }


def numeric_delta(current: dict[str, Any], previous: dict[str, Any], key: str) -> float | None:
    current_value = current.get(key)
    previous_value = previous.get(key)
    if not isinstance(current_value, (int, float)) or not isinstance(previous_value, (int, float)):
        return None
    return float(current_value) - float(previous_value)


def compare_timeframe(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    numeric_keys = [
        "close", "ema50", "ema200", "ema_spread_pct", "pdi", "mdi", "adx",
        "atr14", "atr_pct",
    ]
    return {
        "last_complete_candle_end_previous": previous.get("last_complete_candle_end"),
        "last_complete_candle_end_current": current.get("last_complete_candle_end"),
        "new_complete_candle": current.get("last_complete_candle_end") != previous.get("last_complete_candle_end"),
        "numeric_deltas": {key: numeric_delta(current, previous, key) for key in numeric_keys},
        "obv_relation_previous": previous.get("obv_relation"),
        "obv_relation_current": current.get("obv_relation"),
        "obv_direction_previous": previous.get("obv_direction"),
        "obv_direction_current": current.get("obv_direction"),
        "evidence_score_previous": (previous.get("evidence") or {}).get("score"),
        "evidence_score_current": (current.get("evidence") or {}).get("score"),
    }


def controls(main_price: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, url, parser in [
        ("coinbase", "https://api.exchange.coinbase.com/products/BTC-USD/ticker", lambda p: (float(p["price"]), p.get("time"))),
        ("coingecko", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true&include_last_updated_at=true", lambda p: (float(p["bitcoin"]["usd"]), datetime.fromtimestamp(int(p["bitcoin"]["last_updated_at"]), timezone.utc).isoformat())),
    ]:
        try:
            payload, meta = get_json(url)
            price, source_time = parser(payload)
            diff = (price / main_price - 1) * 100
            result[name] = {"status": "A", "price": price, "source_time": source_time, "obtained_at": meta["obtained_at"], "difference_from_main_pct": diff, "comparison": "MAJOR_SOURCE_CONFLICT" if abs(diff) > 2 else "WARN" if abs(diff) > 0.5 else "PASS"}
        except Exception as exc:
            result[name] = {"status": "N", "error": str(exc)}
    return result


def main() -> int:
    local = datetime.now(ZoneInfo(TZ_NAME))
    run_id = f"run-btc-radar-{local.strftime('%Y%m%dT%H%M%S')}{local.strftime('%Z')}"
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    errors: list[dict[str, Any]] = []
    bundle = None

    try:
        bundle = fetch_pionex(now_ms)
    except CollectorError as exc:
        errors.append(exc.record("PIONEX"))

    if bundle is None:
        try:
            bundle, host_errors = fetch_binance(now_ms)
            errors.extend(host_errors)
        except CollectorError as exc:
            if isinstance(exc.details, list):
                errors.extend(exc.details)
            errors.append(exc.record("BINANCE"))

    schema_version = "btc-radar-input-0.2.0"

    if bundle is None:
        latest = {
            "schema_version": schema_version,
            "run_id": run_id,
            "status": "FAILED",
            "generated_at": local.isoformat(),
            "source": None,
            "indicators": {"4h": None, "1d": None},
            "classification": None,
            "baseline": {"status": "NOT_UPDATED", "baseline_updated": False},
            "quality_gate": "FAIL",
            "information_risk": "HIGH",
            "errors": errors,
            "safety": {"read_only": True, "execution_allowed": False, "human_gate_required": True},
        }
        atomic_json(LATEST, latest)
    else:
        try:
            h4, d1 = indicators(bundle["4h"]), indicators(bundle["1d"])
            classification = classify(h4, d1)
            raw_pack = {
                "run_id": run_id,
                "generated_at": local.isoformat(),
                "source": {k: bundle[k] for k in ("venue", "role", "qualification", "hostname", "instrument", "market")},
                "ticker": bundle["ticker"],
                "candles_4h": bundle["4h"].to_dict("records"),
                "candles_1d": bundle["1d"].to_dict("records"),
                "acquisition_meta": bundle["meta"],
            }
            raw_hash = sha256(raw_pack)
            raw_path = RAW / f"{run_id}.json"
            atomic_json(raw_path, raw_pack)

            snapshot = {
                "run_id": run_id,
                "source": bundle["venue"],
                "instrument": bundle["instrument"],
                "market": bundle["market"],
                "method_id": METHOD_ID,
                "method_version": METHOD_VERSION,
                "ticker_price": bundle["ticker"]["price"],
                "4h": h4,
                "1d": d1,
                "classification": classification,
            }

            baseline_filename = f"{bundle['venue'].lower()}_{bundle['instrument'].lower()}_{METHOD_ID.lower()}.json"
            baseline_path = BASELINES / baseline_filename
            previous = load_json(baseline_path, None)
            comparable = bool(
                previous
                and all(
                    previous.get(k) == snapshot.get(k)
                    for k in ("source", "instrument", "market", "method_id", "method_version")
                )
            )

            baseline: dict[str, Any] = {
                "status": "COMPARABLE" if comparable else "NO_COMPARABLE_BASELINE",
                "baseline_run_id": previous.get("run_id") if comparable else None,
                "baseline_updated": False,
                "baseline_path": str(baseline_path.relative_to(ROOT)).replace("\\", "/"),
            }

            should_update_baseline = not comparable
            if comparable:
                data_cut_changed = (
                    h4.get("last_complete_candle_end") != previous.get("4h", {}).get("last_complete_candle_end")
                    or d1.get("last_complete_candle_end") != previous.get("1d", {}).get("last_complete_candle_end")
                )
                baseline["data_cut_changed"] = data_cut_changed
                baseline["changes"] = {
                    "ticker_price_delta_pct": (snapshot["ticker_price"] / previous["ticker_price"] - 1) * 100,
                    "classification_previous": previous["classification"],
                    "classification_current": classification,
                    "4h": compare_timeframe(h4, previous["4h"]),
                    "1d": compare_timeframe(d1, previous["1d"]),
                }
                should_update_baseline = data_cut_changed
                baseline["update_reason"] = "NEW_COMPLETE_CANDLE" if data_cut_changed else "NO_NEW_COMPLETE_CANDLE"
            else:
                baseline["update_reason"] = "NEW_METHOD_OR_SOURCE_BASELINE"

            if should_update_baseline:
                atomic_json(baseline_path, snapshot)
                baseline["baseline_updated"] = True
                baseline["current_baseline_run_id"] = run_id
            else:
                baseline["current_baseline_run_id"] = previous.get("run_id") if previous else None

            control = controls(bundle["ticker"]["price"])
            major = any(x.get("comparison") == "MAJOR_SOURCE_CONFLICT" for x in control.values())
            latest = {
                "schema_version": schema_version,
                "run_id": run_id,
                "status": "COMPLETE",
                "generated_at": local.isoformat(),
                "timezone": TZ_NAME,
                "method": {
                    "id": METHOD_ID,
                    "version": METHOD_VERSION,
                    "parameters": {
                        "ema": [50, 200],
                        "dmi_period": 14,
                        "adx_smoothing": 6,
                        "atr_period": 14,
                        "maobv_period": 30,
                        "completed_candles_only": True,
                        "requested_candles": CANDLE_LIMIT,
                        "timeframe_alignment_policy": "same_non_neutral_evidence_bias",
                        "baseline_update_policy": "new_complete_candle_only",
                    },
                },
                "source": {
                    "venue": bundle["venue"],
                    "role": bundle["role"],
                    "qualification_status": bundle["qualification"],
                    "hostname": bundle["hostname"],
                    "instrument": bundle["instrument"],
                    "market": bundle["market"],
                    "availability": "A",
                },
                "ticker": bundle["ticker"],
                "indicators": {"4h": h4, "1d": d1},
                "classification": classification,
                "controls": control,
                "baseline": baseline,
                "raw_pack": {
                    "path": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": raw_hash,
                },
                "quality_gate": "FAIL" if major else "WARN",
                "information_risk": "HIGH" if major else "MEDIUM",
                "errors": errors,
                "safety": {"read_only": True, "execution_allowed": False, "human_gate_required": True},
            }
            atomic_json(LATEST, latest)
        except Exception as exc:
            errors.append({
                "source": bundle.get("venue"),
                "error_class": "CALCULATION_FAILURE",
                "stage": "CALCULATION_OR_WRITE",
                "message": f"{type(exc).__name__}: {exc}",
                "details": {},
            })
            latest = {
                "schema_version": schema_version,
                "run_id": run_id,
                "status": "FAILED",
                "generated_at": local.isoformat(),
                "source": {"venue": bundle.get("venue")},
                "indicators": {"4h": None, "1d": None},
                "classification": None,
                "baseline": {"status": "NOT_UPDATED", "baseline_updated": False},
                "quality_gate": "FAIL",
                "information_risk": "HIGH",
                "errors": errors,
                "safety": {"read_only": True, "execution_allowed": False, "human_gate_required": True},
            }
            atomic_json(LATEST, latest)

    log = load_json(RUN_LOG, [])
    log = log if isinstance(log, list) else []
    log.append({
        "run_id": latest["run_id"],
        "generated_at": latest["generated_at"],
        "status": latest["status"],
        "source": (latest.get("source") or {}).get("venue"),
        "direction": (latest.get("classification") or {}).get("direction"),
        "character": (latest.get("classification") or {}).get("character"),
        "signal_strength": (latest.get("classification") or {}).get("signal_strength"),
        "timeframe_tension": (latest.get("classification") or {}).get("timeframe_tension"),
        "quality_gate": latest["quality_gate"],
        "information_risk": latest["information_risk"],
    })
    atomic_json(RUN_LOG, log[-30:])
    RAW.mkdir(parents=True, exist_ok=True)
    for old in sorted(RAW.glob("run-btc-radar-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[30:]:
        old.unlink(missing_ok=True)

    print(json.dumps({
        "run_id": latest["run_id"],
        "status": latest["status"],
        "source": (latest.get("source") or {}).get("venue"),
        "classification": latest.get("classification"),
        "quality_gate": latest["quality_gate"],
        "information_risk": latest["information_risk"],
    }, ensure_ascii=False, indent=2))
    return 0 if latest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())
