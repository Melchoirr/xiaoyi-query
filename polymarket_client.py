"""Polymarket API client: fetch events, markets, and price curves."""

import time
import json
import requests
from config import (
    POLYMARKET_EVENTS_URL, POLYMARKET_MARKETS_URL,
    POLYMARKET_TIMESERIES_URL, POLYMARKET_RATE_LIMIT,
    EVENTS_FILE, PRICE_CURVES_FILE
)


def fetch_events(limit=100, closed=False, tag=None):
    """Fetch active events from Polymarket.

    Returns list of event dicts with keys: id, title, description, category, tags, markets.
    """
    params = {"closed": str(closed).lower(), "limit": limit}
    if tag:
        params["tag"] = tag

    print(f"[Polymarket] Fetching events (limit={limit}, closed={closed})...")
    resp = requests.get(POLYMARKET_EVENTS_URL, params=params, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    # API wraps in list if single event, check structure
    if isinstance(events, dict):
        events = events.get("events", [events])
    print(f"[Polymarket] Got {len(events)} events")
    return events


def fetch_markets(event_id):
    """Fetch all markets for a given event."""
    time.sleep(POLYMARKET_RATE_LIMIT)
    params = {"event_id": event_id}
    resp = requests.get(POLYMARKET_MARKETS_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_price_history(market_id, start_ts=None, end_ts=None, interval="1h"):
    """Fetch price history (timeseries) for a market.

    Returns list of {timestamp, price} dicts.
    """
    time.sleep(POLYMARKET_RATE_LIMIT * 0.5)  # shorter wait since this can be many
    params = {
        "market": market_id,
        "interval": interval,
        "fidelity": "clob",
    }
    if start_ts:
        params["startTs"] = start_ts
    if end_ts:
        params["endTs"] = end_ts

    resp = requests.get(POLYMARKET_TIMESERIES_URL, params=params, timeout=30)
    if resp.status_code == 429:
        print(f"  [Polymarket] Rate limited on market {market_id}, waiting 5s...")
        time.sleep(5)
        resp = requests.get(POLYMARKET_TIMESERIES_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_polymarket_pipeline(event_limit=50):
    """Full Polymarket pipeline: fetch events + markets + price curves.
    Saves to EVENTS_FILE and PRICE_CURVES_FILE.
    """
    events = fetch_events(limit=event_limit)

    events_data = []
    curves_data = []

    for i, event in enumerate(events):
        event_id = event.get("id") or event.get("slug")
        title = event.get("title") or event.get("question", "")
        if not event_id:
            continue

        # Build event record
        record = {
            "event_id": event_id,
            "title": title,
            "description": event.get("description", ""),
            "category": event.get("category", ""),
            "tags": [t.get("label", t) if isinstance(t, dict) else t
                      for t in event.get("tags", [])],
            "volume": event.get("volume", 0),
            "liquidity": event.get("liquidity", 0),
            "end_date": event.get("endDate") or event.get("end_date", ""),
            "markets": [],
        }
        events_data.append(record)

        # Fetch markets
        print(f"  [{i+1}/{len(events)}] {title[:80]}")
        try:
            markets = fetch_markets(event_id)
        except Exception as e:
            print(f"    Markets error: {e}")
            continue

        if not isinstance(markets, list):
            markets = [markets]

        for m in markets:
            m_id = m.get("id") or m.get("conditionId")
            question = m.get("question", "")
            outcomes = m.get("outcomes", [])
            outcome_prices = m.get("outcomePrices", [])

            record["markets"].append({
                "market_id": m_id,
                "question": question,
                "outcomes": [o.get("label", o) if isinstance(o, dict) else o
                             for o in outcomes],
                "current_prices": outcome_prices,
            })

            # Fetch price history
            if m_id:
                history = []
                try:
                    raw = fetch_price_history(m_id)
                    history = raw.get("history", raw) if isinstance(raw, dict) else raw
                except Exception as e:
                    print(f"    Price history error for {m_id}: {e}")

                if history:
                    curves_data.append({
                        "market_id": m_id,
                        "event_id": event_id,
                        "question": question,
                        "history": history,
                    })

    # Save
    with open(EVENTS_FILE, "w") as f:
        json.dump(events_data, f, ensure_ascii=False, indent=2)
    print(f"[Polymarket] Saved {len(events_data)} events to {EVENTS_FILE}")

    with open(PRICE_CURVES_FILE, "w") as f:
        json.dump(curves_data, f, ensure_ascii=False, indent=2)
    print(f"[Polymarket] Saved {len(curves_data)} price curves to {PRICE_CURVES_FILE}")

    return events_data, curves_data


if __name__ == "__main__":
    run_polymarket_pipeline()
