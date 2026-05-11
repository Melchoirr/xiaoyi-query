#!/usr/bin/env python3
"""Main pipeline: Polymarket → Wikidata → Knowledge Graph.

Usage:
    python main.py [--event-limit N] [--no-expand] [--llm-key KEY]

Flags:
    --event-limit N    Max events to fetch (default: 20)
    --no-expand        Skip Wikidata relation expansion (faster)
    --llm-key KEY      DeepSeek API key for LLM enrichment (optional)
    --llm-model MODEL  Model name (default: deepseek-chat)
"""

import argparse
import sys
from polymarket_client import run_polymarket_pipeline
from entity_linker import run_entity_linking
from knowledge_graph import build_graph, KnowledgeGraphDB
from gdelt_matcher import RealGDELTMatcher
from llm_enricher import LLMEnricher, merge_llm_relations


def main():
    parser = argparse.ArgumentParser(description="Polymarket → Knowledge Graph pipeline")
    parser.add_argument("--event-limit", type=int, default=20,
                        help="Max events to fetch from Polymarket")
    parser.add_argument("--no-expand", action="store_true",
                        help="Skip Wikidata relation expansion")
    parser.add_argument("--llm-key", type=str, default=None,
                        help="DeepSeek API key (also set via DEEPSEEK_API_KEY env)")
    parser.add_argument("--llm-model", type=str, default="deepseek-chat",
                        help="LLM model name")
    args = parser.parse_args()

    # Step 1: Fetch Polymarket data
    print("=" * 60)
    print("STEP 1: Polymarket events + price curves")
    print("=" * 60)
    events, curves = run_polymarket_pipeline(event_limit=args.event_limit)
    if not events:
        print("No events fetched. Exiting.")
        sys.exit(1)

    # Step 2: Entity extraction + Wikidata linking
    print("\n" + "=" * 60)
    print("STEP 2: Entity extraction + Wikidata linking")
    print("=" * 60)
    entity_results = run_entity_linking(events, expand=not args.no_expand)

    # Step 3: Build knowledge graph
    print("\n" + "=" * 60)
    print("STEP 3: Build knowledge graph (Wikidata)")
    print("=" * 60)
    graph = build_graph(entity_results, events, curves)
    print(f"  Nodes: {graph['stats']['total_nodes']}")
    print(f"  Edges: {graph['stats']['total_edges']}")
    print(f"  Events indexed: {graph['stats']['total_events']}")

    # Step 4: LLM enrichment (customers, competitors, partnerships)
    print("\n" + "=" * 60)
    print("STEP 4: LLM enrichment (customers / competitors / partners)")
    print("=" * 60)
    enricher = LLMEnricher(api_key=args.llm_key, model=args.llm_model)
    if enricher.is_available:
        graph = merge_llm_relations(graph, enricher)
    else:
        print("  No API key provided, skipping. Set --llm-key or DEEPSEEK_API_KEY env.")
        print("  (Dry-run mode: code is ready, no API call made)")

    # Step 5: SQLite database
    print("\n" + "=" * 60)
    print("STEP 5: SQLite database")
    print("=" * 60)
    graph["_events_raw"] = events  # temp reference for DB loading
    db = KnowledgeGraphDB()
    db.load_from_graph(graph)
    print("  SQLite DB ready")

    # Step 6: Real GDELT matcher
    print("\n" + "=" * 60)
    print("STEP 6: GDELT matcher (DuckDB/KG-based)")
    print("=" * 60)
    matcher = RealGDELTMatcher(graph)
    for evt_id in list(graph["event_entity_index"].keys())[:3]:
        result = matcher.match_news(evt_id)
        entities = result["matched_entities"]
        articles = result["articles"]
        print(f"  Event: {evt_id}")
        print(f"    Search terms: {result['search_terms'][:8]}")
        print(f"    Articles matched: {result['total_matched']:,}")
        print(f"    Core entities: {', '.join(e['label'] + ' (' + e['qid'] + ')' for e in entities)}")
    matcher.close()

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Output files in data/")


if __name__ == "__main__":
    main()
