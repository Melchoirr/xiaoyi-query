"""Run the knowledge graph pipeline using cached event data (skip Polymarket API).
Then match against GDELT news via DuckDB.
"""

import json
import sys
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, str(Path(__file__).parent))

from entity_linker import run_entity_linking
from knowledge_graph import build_graph, KnowledgeGraphDB
from gdelt_matcher import RealGDELTMatcher
from config import DATA_DIR

# Event data from the existing summary (10 events known to have GDELT data)
EVENTS = [
    {"event_id": "236884", "title": "Iran x Israel/US conflict ends by...?",
     "description": "Will the Iran-Israel/US conflict end by March 31?", "category": "politics", "tags": ["iran", "israel", "us", "conflict"]},
    {"event_id": "35908", "title": "Who will Trump nominate as Fed Chair?",
     "description": "Will Trump nominate Scott Bessent as the next Fed chair?", "category": "politics", "tags": ["trump", "fed", "nomination"]},
    {"event_id": "73130", "title": "Will the U.S. invade Iran before 2027?",
     "description": "Will the United States invade Iran before 2027?", "category": "politics", "tags": ["us", "iran", "invasion"]},
    {"event_id": "118172", "title": "Will Trump acquire Greenland before 2027?",
     "description": "Will Trump acquire Greenland before 2027?", "category": "politics", "tags": ["trump", "greenland"]},
    {"event_id": "34044", "title": "Will China invade Taiwan by end of 2026?",
     "description": "Will China invade Taiwan by end of 2026?", "category": "politics", "tags": ["china", "taiwan", "invasion"]},
    {"event_id": "34050", "title": "Russia x Ukraine ceasefire by end of 2026?",
     "description": "Will Russia and Ukraine reach a ceasefire by end of 2026?", "category": "politics", "tags": ["russia", "ukraine", "ceasefire"]},
    {"event_id": "67284", "title": "Fed decision in March?",
     "description": "Will the Fed decrease interest rates by 25 bps after the March 2026 meeting?", "category": "politics", "tags": ["fed", "interest rates"]},
    {"event_id": "237306", "title": "Will Iran strike gulf oil facilities by March 31?",
     "description": "Will Iran strike gulf oil facilities by March 31?", "category": "politics", "tags": ["iran", "oil", "gulf"]},
    {"event_id": "257313", "title": "US-Iran nuclear deal by April 30?",
     "description": "Will US and Iran reach a nuclear deal by April 30?", "category": "politics", "tags": ["us", "iran", "nuclear"]},
    {"event_id": "75478", "title": "Fed decision in April?",
     "description": "Will the Fed increase interest rates by 25+ bps after the April 2026 meeting?", "category": "politics", "tags": ["fed", "interest rates"]},
]

# Fake price curves (we already have the merged data, just need KG structure)
# For real price data, we'll use the existing CSVs later
PRICE_CURVES = []


def main():
    print("=" * 60)
    print("STEP 2: Entity extraction + Wikidata linking")
    print("=" * 60)
    entity_results = run_entity_linking(EVENTS, expand=True)

    print("\n" + "=" * 60)
    print("STEP 3: Build knowledge graph")
    print("=" * 60)
    graph = build_graph(entity_results, EVENTS, PRICE_CURVES)
    print(f"  Nodes: {graph['stats']['total_nodes']}")
    print(f"  Edges: {graph['stats']['total_edges']}")
    print(f"  Events indexed: {graph['stats']['total_events']}")

    # Save events for DB loading
    graph["_events_raw"] = EVENTS

    print("\n" + "=" * 60)
    print("STEP 5: SQLite database")
    print("=" * 60)
    db = KnowledgeGraphDB()
    db.load_from_graph(graph)
    print("  SQLite DB ready")

    # Print entity summary per event
    print("\n" + "=" * 60)
    print("Entity linking results per event:")
    print("=" * 60)
    for eid, qids in graph["event_entity_index"].items():
        titles = [graph["nodes"][i] if isinstance(i, int) else
                  next((n for n in graph["nodes"] if n["qid"] == i), None)
                  for i in qids]
        event_info = next((e for e in EVENTS if e["event_id"] == eid), None)
        if event_info:
            labels = []
            for qid in qids:
                node = next((n for n in graph["nodes"] if n["qid"] == qid), None)
                if node:
                    labels.append(f"{node['label']} ({qid})")
            print(f"  {eid}: {event_info['title'][:60]}")
            print(f"    Entities: {', '.join(labels[:5])}")

    print("\n" + "=" * 60)
    print("STEP 6: Real GDELT matcher demo")
    print("=" * 60)
    matcher = RealGDELTMatcher(graph)
    for evt_id in sorted(graph["event_entity_index"].keys())[:3]:
        result = matcher.match_news(evt_id)
        print(f"  Event: {evt_id}")
        print(f"    Search terms: {result['search_terms'][:10]}")
        print(f"    Articles matched: {result['total_matched']}")
    matcher.close()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Output files in data/")


if __name__ == "__main__":
    main()
