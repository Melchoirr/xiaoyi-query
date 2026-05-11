"""Expand Polymarket event pool: fetch diverse events, run entity linking,
build KG, run V2 GDELT alignment, generate plots.

Strategy:
- Fetch 100 open events (high volume, active trading)
- Filter: volume > $500K, news-relevant tags (Politics, Geopolitics, Crypto, Finance, Tech, World)
- Exclude pure sports events (GDELT doesn't cover game recaps well)
- Run full pipeline: entity linking → KG → V2 alignment → plots
"""

import json
import sys
import time
from pathlib import Path
import requests
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    POLYMARKET_EVENTS_URL, POLYMARKET_MARKETS_URL, POLYMARKET_TIMESERIES_URL,
    POLYMARKET_RATE_LIMIT, DATA_DIR, EVENTS_FILE, PRICE_CURVES_FILE, ENTITIES_FILE
)
from entity_linker import run_entity_linking
from knowledge_graph import build_graph
from gdelt_matcher_v2 import EnhancedGDELTMatcher, run_enhanced_alignment

# ---------------------------------------------------------------------------
# Event selection config
# ---------------------------------------------------------------------------
MIN_VOLUME = 500_000          # Minimum USD volume
NEWS_RELEVANT_TAGS = {        # Tags that indicate GDELT-worthy events
    "Politics", "Geopolitics", "World", "Crypto", "Finance", "Economy",
    "Elections", "Global Elections", "US Election", "World Elections",
    "Trump Presidency", "Trump", "Ukraine", "Israel", "Middle East",
    "Foreign Policy", "Tech", "AI", "Business", "Breaking News",
    "Pre-Market", "Airdrops", "Crypto Prices",
}
SPORTS_TAGS = {               # Exclude pure sports
    "Sports", "Soccer", "NBA", "NHL", "NBA Finals", "Basketball",
    "Hockey", "NFL", "MLB", "UFC", "F1", "Tennis", "Golf",
    "EPL", "Champions League", "La Liga", "Ligue 1", "Serie A",
    "UEFA", "Stanley Cup", "NBA Playoffs", "MVP",
}

# Already processed in V1/V2
EXISTING_IDS = {"236884", "35908", "73130", "118172", "34044", "34050",
                "67284", "237306", "257313", "75478"}


def fetch_open_events(limit=100):
    """Fetch open events from Polymarket."""
    params = {"closed": "false", "limit": limit}
    print(f"[Polymarket] Fetching up to {limit} open events...")
    resp = requests.get(POLYMARKET_EVENTS_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        events = data.get("events", [])
    else:
        events = data
    print(f"  Retrieved {len(events)} events")
    return events


def is_news_relevant(event):
    """Check if an event's tags suggest GDELT news coverage."""
    tags = {t.get("label", t) if isinstance(t, dict) else t
            for t in (event.get("tags") or [])}
    # Exclude sports-heavy events
    if tags & SPORTS_TAGS:
        return False
    # Include if any tag matches news-relevant set
    return bool(tags & NEWS_RELEVANT_TAGS)


def select_events(events, max_events=60):
    """Select diverse, high-volume, news-relevant events."""
    # Filter
    candidates = []
    for e in events:
        eid = str(e.get("id", ""))
        if eid in EXISTING_IDS:
            continue
        vol = e.get("volume", 0) or 0
        if vol < MIN_VOLUME:
            continue
        if not is_news_relevant(e):
            continue
        candidates.append(e)

    # Sort by volume descending, take top N
    candidates.sort(key=lambda e: e.get("volume", 0) or 0, reverse=True)

    selected = candidates[:max_events]

    print(f"\n  Filtered {len(candidates)} news-relevant events with vol > ${MIN_VOLUME/1e6:.1f}M")
    print(f"  Selected top {len(selected)} for pipeline")

    # Show selection summary
    tag_counts = {}
    for e in selected:
        for t in (e.get("tags") or []):
            tn = t.get("label", t) if isinstance(t, dict) else t
            if tn in NEWS_RELEVANT_TAGS:
                tag_counts[tn] = tag_counts.get(tn, 0) + 1

    print(f"  Tag coverage: {dict(sorted(tag_counts.items(), key=lambda x: -x[1])[:10])}")

    return selected


def fetch_price_curves(events):
    """Fetch price history for all markets of selected events."""
    curves_data = []

    for i, event in enumerate(events):
        event_id = str(event.get("id", ""))
        title = event.get("title", "")[:80]
        print(f"  [{i+1}/{len(events)}] {title}")

        time.sleep(POLYMARKET_RATE_LIMIT)
        try:
            resp = requests.get(POLYMARKET_MARKETS_URL,
                              params={"event_id": event_id}, timeout=30)
            resp.raise_for_status()
            markets = resp.json()
        except Exception as e:
            print(f"    Markets error: {e}")
            continue

        if not isinstance(markets, list):
            markets = [markets]

        for m in markets:
            m_id = m.get("id") or m.get("conditionId")
            if not m_id:
                continue

            time.sleep(POLYMARKET_RATE_LIMIT * 0.5)
            try:
                resp = requests.get(POLYMARKET_TIMESERIES_URL,
                                  params={"market": m_id, "interval": "1h", "fidelity": "clob"},
                                  timeout=30)
                if resp.status_code == 429:
                    time.sleep(5)
                    resp = requests.get(POLYMARKET_TIMESERIES_URL,
                                      params={"market": m_id, "interval": "1h", "fidelity": "clob"},
                                      timeout=30)
                resp.raise_for_status()
                raw = resp.json()
                history = raw.get("history", raw) if isinstance(raw, dict) else raw
            except Exception as e:
                print(f"    Price history error for {m_id}: {e}")
                continue

            if history:
                curves_data.append({
                    "market_id": m_id,
                    "event_id": event_id,
                    "question": m.get("question", ""),
                    "history": history,
                })

    print(f"  Fetched price curves for {len(curves_data)} markets")
    return curves_data


def build_event_records(selected_events):
    """Convert Polymarket event dicts to pipeline-compatible records."""
    records = []
    for e in selected_events:
        records.append({
            "event_id": str(e.get("id", "")),
            "title": e.get("title", ""),
            "description": e.get("description", ""),
            "category": e.get("category", ""),
            "tags": [t.get("label", t) if isinstance(t, dict) else t
                     for t in e.get("tags", [])],
            "volume": e.get("volume", 0) or 0,
            "end_date": e.get("endDate", ""),
        })
    return records


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=50,
                       help="Maximum events to process (default: 50)")
    parser.add_argument("--sim-threshold", type=float, default=0.35,
                       help="Embedding similarity threshold (default: 0.35)")
    parser.add_argument("--output-dir", type=str,
                       default="result/by_date/2026-05-11/kg_gdelt_alignment_v3",
                       help="Output directory for results")
    parser.add_argument("--skip-fetch", action="store_true",
                       help="Skip Polymarket API fetch, use cached events.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Phase 1: Fetch & select events
    # =========================================================================
    print("=" * 70)
    print("PHASE 1: Event Selection")
    print("=" * 70)

    curves = []  # price curves, used later in Phase 3c

    if args.skip_fetch and EVENTS_FILE.exists():
        with open(EVENTS_FILE) as f:
            selected_events = json.load(f)
        print(f"  Loaded {len(selected_events)} cached events from {EVENTS_FILE}")
        # Also load cached price curves if available
        if PRICE_CURVES_FILE.exists():
            with open(PRICE_CURVES_FILE) as f:
                curves = json.load(f)
            print(f"  Loaded {len(curves)} cached price curves from {PRICE_CURVES_FILE}")
    else:
        all_events = fetch_open_events(limit=100)
        selected_events = select_events(all_events, max_events=args.max_events)

        # Save expanded events
        event_records = build_event_records(selected_events)
        with open(EVENTS_FILE, "w") as f:
            json.dump(event_records, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(event_records)} events to {EVENTS_FILE}")

        # Save price curves
        print(f"\n  Fetching price curves...")
        curves = fetch_price_curves(selected_events)
        with open(PRICE_CURVES_FILE, "w") as f:
            json.dump(curves, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(curves)} price curves to {PRICE_CURVES_FILE}")

    # =========================================================================
    # Phase 2: Entity linking + KG build
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: Entity Linking + Knowledge Graph")
    print("=" * 70)

    # If loaded from cache, these are already converted records.
    if selected_events and "id" in selected_events[0] and "event_id" not in selected_events[0]:
        event_records = build_event_records(selected_events)
    else:
        event_records = selected_events
    print(f"  Processing {len(event_records)} events for entity linking...")

    entity_results = run_entity_linking(event_records, expand=True)

    # Build KG
    graph = build_graph(entity_results, event_records, [])
    print(f"  KG: {graph['stats']['total_nodes']} nodes, "
          f"{graph['stats']['total_edges']} edges, "
          f"{graph['stats']['total_events']} events")

    # Save KG
    with open(DATA_DIR / "knowledge_graph_v3.json", "w") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # Phase 3: V2 GDELT alignment
    # =========================================================================
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: V2 GDELT Alignment (P0+P1+P2)")
    print("=" * 70)

    graph["_events_raw"] = event_records

    matcher = EnhancedGDELTMatcher(graph, sim_threshold=args.sim_threshold)

    # Phase 3a: Match each event
    phase1_results = {}
    for event_id in sorted(graph["event_entity_index"].keys()):
        event_title = ""
        for evt in event_records:
            if evt.get("event_id") == event_id:
                event_title = evt.get("title", event_id)
                break

        print(f"\n  Event {event_id}: {event_title[:70]}")
        result = matcher.match_event(event_id)
        phase1_results[event_id] = result

        print(f"    Search terms: {result['search_terms'][:8]}...")
        print(f"    Retrieved: {result['total_retrieved']:,}")
        print(f"    After embedding filter: {result['total_filtered']:,}")
        avg_sim = np.mean([a["similarity"] for a in result["articles"]]) if result["articles"] else 0
        print(f"    Mean similarity: {avg_sim:.3f}")

    # Phase 3b: P1 dedup
    print(f"\n{'=' * 60}")
    print("P1 DEDUP: Resolving URL assignments")
    print("=" * 60)
    per_event_raw = {eid: r["articles"] for eid, r in phase1_results.items()}
    per_event_dedup = matcher.resolve_dedup(per_event_raw)

    total_removed = 0
    for event_id in sorted(per_event_dedup.keys()):
        raw_n = len(per_event_raw[event_id])
        dedup_n = len(per_event_dedup[event_id])
        removed = raw_n - dedup_n
        total_removed += removed
        if removed > 0:
            print(f"  {event_id}: {raw_n:,} → {dedup_n:,} ({removed:,} removed)")
    print(f"  Total removed by dedup: {total_removed:,}")

    # Phase 3c: Save merged series
    print(f"\n{'=' * 70}")
    print("PHASE 4: Save merged series + plots")
    print("=" * 70)

    from plot_alignment_v2 import plot_event

    results = []
    for event_id in sorted(per_event_dedup.keys()):
        articles = per_event_dedup[event_id]
        hourly = matcher.compute_hourly_series(articles)

        event_title = ""
        for evt in event_records:
            if evt.get("event_id") == event_id:
                event_title = evt.get("title", event_id)
                break

        safe_name = event_title.lower().replace(" ", "-")[:80]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "" for c in safe_name)

        # Build hourly price from price curves
        price_rows = {}
        for curve in curves:
            if str(curve.get("event_id")) == str(event_id):
                for pt in curve.get("history", []):
                    ts = pt.get("t", pt.get("timestamp", ""))
                    p = pt.get("p", pt.get("price", 0))
                    if ts and p:
                        try:
                            dt = pd.to_datetime(ts, unit='s', utc=True).floor('h')
                        except Exception:
                            dt = pd.to_datetime(ts, utc=True).floor('h')
                        if dt not in price_rows:
                            price_rows[dt] = []
                        price_rows[dt].append(float(p))

        if price_rows:
            price_hourly = pd.DataFrame([
                {"datetime_utc": dt, "price": np.mean(prices)}
                for dt, prices in sorted(price_rows.items())
            ])
        else:
            price_hourly = pd.DataFrame(columns=["datetime_utc", "price"])

        # Merge price + news
        if not hourly.empty and not price_hourly.empty:
            price_hourly["hour"] = price_hourly["datetime_utc"].dt.tz_localize(None).dt.floor("h")
            hourly["hour"] = hourly["datetime_utc"].dt.tz_localize(None).dt.floor("h")
            merged = pd.merge(price_hourly, hourly, on="hour", how="outer")
            merged["datetime_utc"] = pd.to_datetime(merged["hour"], utc=True)
            merged = merged.drop(columns=["hour"])
            merged = merged.sort_values("datetime_utc")
            merged["news_volume"] = merged["news_volume"].fillna(0).astype(int)
            merged["price"] = merged["price"].interpolate(method="linear", limit_direction="both")
            merged["price"] = merged["price"].ffill().bfill()
        elif not hourly.empty:
            merged = hourly.copy()
            merged["price"] = np.nan
        else:
            merged = pd.DataFrame(columns=["datetime_utc", "price", "news_volume"])

        csv_path = output_dir / f"event_{event_id}_{safe_name}_merged_series.csv"
        merged.to_csv(csv_path, index=False)
        print(f"  {event_id}: {len(merged)} rows → {csv_path.name}")

        # Generate plot (no tone)
        if not merged.empty and "price" in merged.columns:
            png_path = output_dir / f"event_{event_id}_{safe_name}_alignment_v3.png"
            plot_event(merged, event_title, png_path)

        results.append({
            "event_id": event_id,
            "event_title": event_title,
            "csv": str(csv_path),
            "rows": len(merged),
            "news_total": len(articles),
            "search_terms": phase1_results[event_id]["search_terms"],
        })

    # Save summary
    summary_rows = []
    for r in results:
        summary_rows.append({
            "event_id": r["event_id"],
            "event_title": r["event_title"],
            "news_total": r["news_total"],
            "rows": r["rows"],
            "search_terms": ", ".join(r["search_terms"][:5]),
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "alignment_summary.csv", index=False)

    # =========================================================================
    # Print overview
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("EXPANDED DATASET SUMMARY")
    print("=" * 70)
    print(f"  Events processed: {len(results)}")
    print(f"  Output directory: {output_dir}")

    news_totals = [r["news_total"] for r in results]
    if news_totals:
        print(f"  News per event: median={np.median(news_totals):.0f}, "
              f"mean={np.mean(news_totals):.0f}, "
              f"min={np.min(news_totals)}, max={np.max(news_totals)}")
        print(f"  Total news articles (after dedup): {sum(news_totals):,}")

    zero_news = [r for r in results if r["news_total"] == 0]
    if zero_news:
        print(f"  WARNING: {len(zero_news)} events have ZERO news matches:")
        for r in zero_news:
            print(f"    - {r['event_id']}: {r['event_title'][:60]}")

    print(f"\nDone! Output in {output_dir}")


if __name__ == "__main__":
    main()
