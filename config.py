import os
from pathlib import Path

# Polymarket API
POLYMARKET_EVENTS_URL = "https://gamma-api.polymarket.com/events"
POLYMARKET_MARKETS_URL = "https://gamma-api.polymarket.com/markets"
POLYMARKET_TIMESERIES_URL = "https://clob.polymarket.com/prices-history"

# Wikidata API
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Output paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

EVENTS_FILE = DATA_DIR / "events.json"
PRICE_CURVES_FILE = DATA_DIR / "price_curves.json"
ENTITIES_FILE = DATA_DIR / "entities.json"
KNOWLEDGE_GRAPH_FILE = DATA_DIR / "knowledge_graph.json"

# API limits
POLYMARKET_RATE_LIMIT = 2.0  # seconds between requests
WIKIDATA_RATE_LIMIT = 0.3   # seconds between API calls

# Entity types to extract from event titles
ENTITY_TYPE_MAP = {
    "PERSON": "person",
    "NORP": "organization",  # Nationalities, religious, political groups
    "ORG": "organization",
    "GPE": "location",        # Countries, cities, states
    "LOC": "location",
    "EVENT": "event",
}
