# backtest.py — Python 3.13 compatible, robust OHLCV detection + sanitized Pandas feed

import argparse
import backtrader as bt
import pandas as pd
import yfinance as yf


# ---------- Robust Pandas -> Backtrader feed (explicit column mapping) ----------
class CleanPandasData(bt.feeds.PandasData):
    """
    Safe PandasData wrapper for modern pandas/yfinance:
      - expects standardized columns: open, high, low, close, volume
      - uses index as datetime (tz-naive)
    """
    params = (
        ("datetime", None),   # use index as datetime
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),
    )


def get_data(ticker: str, start: str, end: str) -> CleanPandasData:
    """
    Downloads data via yfinance, auto-detects OHLCV columns (no matter how named),
    coerces to numeric, drops/fills NaNs, standardizes labels, tz-naive index,
    and returns a CleanPandasData feed for Backtrader.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")

    # 1) Normalize column labels to lowercase strings and flatten tuples/MultiIndex
    def _flat(col):
        if isinstance(col, tuple):
            return "_".join(map(str, col))
        return str(col)

    df.columns = [_flat(c).strip().lower() for c in df.columns]

    # 2) Helper to find a column that *means* open/high/low/close/volume
    def find_col(key: str) -> str:
        """
        Returns a column name that maps to `key`.
        Tries exact match, common suffix/prefix patterns, adjusted-close variants, then substring.
        Raises KeyError if not found.
        """
        cols = list(df.columns)

        # Exact match
        for c in cols:
            if c == key:
                return c

        # Common patterns: *_key, *.key, -key, prefix_key, key_suffix, etc.
        for c in cols:
            s = c.lower()
            parts_us = s.split("_")
            parts_dot = s.split(".")
            parts_dash = s.split("-")
            if (
                s.endswith(f"_{key}") or s.endswith(f".{key}") or s.endswith(f"-{key}")
                or parts_us[-1] == key or parts_dot[-1] == key or parts_dash[-1] == key
            ):
                return c

        # For close, accept adjusted close variants
        if key == "close":
            for c in cols:
                if "adj" in c and "close" in c:
                    return c

        # Last resort: any substring
        for c in cols:
            if key in c:
                return c

        raise KeyError(f"Could not find a column for '{key}' in {cols}")

    # 3) Detect required fields
    open_col   = find_col("open")
    high_col   = find_col("high")
    low_col    = find_col("low")
    close_col  = find_col("close")
    volume_col = find_col("volume")

    # 4) Standardize to exactly: open, high, low, close, volume
    df_std = df[[open_col, high_col, low_col, close_col, volume_col]].copy()
    df_std.columns = ["open", "high", "low", "close", "volume"]

    # 5) Ensure DatetimeIndex, sorted, tz-naive
    if not isinstance(df_std.index, pd.DatetimeIndex):
        df_std.index = pd.to_datetime(df_std.index, errors="coerce")
    if df_std.index.tz is not None:
        df_std.index = df_std.index.tz_localize(None)
    df_std = df_std.sort_index()

    # 6) Coerce numeric & clean NaNs
    for col in ["open", "high", "low", "close", "volume"]:
        df_std[col] = pd.to_numeric(df_std[col], errors="coerce")
    df_std["volume"] = df_std["volume"].fillna(0)
    df_std = df_std.dropna(subset=["open", "high", "low", "close"])

    if df_std.empty or df_std["close"].isna().all():
        raise RuntimeError(
            "All rows dropped after cleaning; no usable OHLC data.\n"
            f"Original columns: {list(df.columns)}"
        )

    first, last = df_std.index.min(), df_std.index.max()
    print(f"[Data] {ticker}: {len(df_std)} bars from {first.date()} to {last.date()}")

    return CleanPandasData(dataname=df_std)


# ---------- Simple SMA crossover strategy ----------
class MovingAverageCrossStrategy(bt.Strategy):
    """
    Buy when the short SMA crosses above the long SMA; sell on cross down.
    """
    params = dict(short=50, long=200, size=10)

    def __init__(self):
        self.sma_short = bt.ind.SMA(period=self.p.short)
        self.sma_long = bt.ind.SMA(period=self.p.long)
        self.crossover = bt.ind.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy(size=self.p.size)
        elif self.position and self.crossover < 0:
            self.close()


def run_backtest(ticker: str, start: str, end: str, cash: float, commission: float,
                 short: int, long: int, save_plots: bool):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MovingAverageCrossStrategy, short=short, long=long)

    data_feed = get_data(ticker, start, end)
    cerebro.adddata(data_feed)

    # Optional: avoid volume subplot issues (some feeds have sparse/zero volume)
    data_feed.plotinfo.plotvolume = False

    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=commission)

    print("\n=== CONFIG ===")
    print(f"Ticker={ticker}  Range={start} → {end}  Cash=${cash:,.2f}  Comm={commission}")
    print(f"SMA short={short}  SMA long={long}\n")

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue()}")
    cerebro.run()
    print(f"Final   Portfolio Value: {cerebro.broker.getvalue()}")

    # Try to show a chart; if GUI/NaN issues occur, save PNGs instead.
    try:
        figs = cerebro.plot(style="candlestick")
        # Some environments return nested lists of figures; try saving too if requested
        if save_plots:
            _save_figs(figs)
    except Exception as e:
        print(f"[Plot] GUI plot failed ({e}); saving PNGs instead…")
        figs = cerebro.plot(style="candlestick", iplot=False)
        _save_figs(figs)


def _iter_figs(obj):
    if isinstance(obj, (list, tuple)):
        for sub in obj:
            yield from _iter_figs(sub)
    else:
        yield obj


def _save_figs(figs):
    import matplotlib.pyplot as plt
    for i, f in enumerate(_iter_figs(figs)):
        try:
            f.savefig(f"backtest_plot_{i}.png", dpi=150, bbox_inches="tight")
        except Exception as e:
            print(f"[Plot] Failed to save figure {i}: {e}")
    plt.close('all')
    print("[Plot] Saved plot(s) as backtest_plot_*.png in this folder.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end",   default="2023-01-01")
    parser.add_argument("--cash", type=float, default=10_000)
    parser.add_argument("--commission", type=float, default=0.001)
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long",  type=int, default=200)
    parser.add_argument("--save-plots", action="store_true",
                        help="Save PNGs even if GUI plotting works")
    args = parser.parse_args()

    run_backtest(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        cash=args.cash,
        commission=args.commission,
        short=args.short,
        long=args.long,
        save_plots=args.save_plots,
    )


if __name__ == "__main__":
    main()

