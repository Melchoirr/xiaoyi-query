# Polymarket → Knowledge Graph → GDELT News Alignment

Align Polymarket event price curves with GDELT news articles via a Wikidata-powered knowledge graph.

## Architecture

```
Polymarket API ──→ Entity Extraction ──→ Wikidata Linking ──→ Knowledge Graph
     │                  (spaCy)              (search + SPARQL)      │
     │                                                              │
     └── Price Curves                                               ├── Wikidata edges (founder, product, industry, competitor...)
                                                                    ├── LLM edges (customers, partners, technologies)
                                                                    └── GDELT Matcher (mock → real)
```

## Pipeline Steps

| Step | Description | Output |
|------|-------------|--------|
| 1. Polymarket | Fetch active events + market data + price curves | `data/events.json`, `data/price_curves.json` |
| 2. Entity Linking | spaCy NER + Wikidata search + disambiguation + relation expansion | `data/entities.json` |
| 3. Knowledge Graph | Build graph (nodes/edges) from Wikidata relations + industry-based competitors | `data/knowledge_graph.json` |
| 4. LLM Enrichment | DeepSeek supplements customer/partner/technology relations Wikidata misses | (optional) |
| 5. SQLite DB | Load graph into queryable database | `data/knowledge_graph.db` |
| 6. GDELT Matcher | Match news to events via entity overlap (mock until GDELT data arrives) | — |

## Setup

```bash
# Conda environment
conda create -n polymarket-alignment python=3.10
conda activate polymarket-alignment

# Dependencies
pip install spacy sparqlwrapper requests
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Basic run (20 events, Wikidata only)
python main.py --event-limit 20

# Skip relation expansion (faster, fewer API calls)
python main.py --event-limit 20 --no-expand

# With LLM enrichment (DeepSeek)
python main.py --event-limit 20 --llm-key sk-xxxxx

# Environment variable alternative
export DEEPSEEK_API_KEY=sk-xxxxx
python main.py --event-limit 20
```

## Knowledge Sources

| Layer | Source | Examples |
|-------|--------|----------|
| Wikidata direct | Founder, CEO, products, industry, location | Jensen Huang, Santa Clara, S&P 500 |
| Wikidata industry | Same-industry companies (competitors) | AMD, Intel, Qualcomm, TSMC |
| Wikidata inverse | Products developed by this entity | CUDA, GeForce, PhysX, DLSS |
| LLM (DeepSeek) | Customers, strategic partners, technologies | Microsoft, Meta, OpenAI |

## NVIDIA Example

For an event mentioning NVIDIA, the knowledge graph expands to:

```
NVIDIA
├── founder/CEO: Jensen Huang ✓ (Wikidata)
├── products: CUDA, GeForce, DLSS ✓ (Wikidata P178)
├── competitors: AMD, Intel, Qualcomm ✓ (Wikidata industry)
├── customers: Microsoft, Meta, OpenAI, Google ✓ (LLM)
└── partners: TSMC, Dell ✓ (LLM)
```

This means a GDELT news article about "Microsoft increases AI infrastructure spending" will match an NVIDIA-related Polymarket event, even though the article never mentions "NVIDIA".

## GDELT Integration

When GDELT GKG data is available, place it on this machine and implement `GDELTMatcher.load_gkg()`. The GKG V2 format includes Wikidata IDs in columns:
- `V2Persons` — person Wikidata IDs
- `V2Organizations` — organization Wikidata IDs  
- `V2Locations` — location Wikidata IDs

The matcher will filter GKG rows by knowledge graph entity Q-IDs and rank by relevance score.

## Output Files

```
data/
├── events.json           # Polymarket events + markets
├── price_curves.json     # Price time series
├── entities.json         # NER + Wikidata linking results
├── knowledge_graph.json  # Full graph (nodes + edges)
└── knowledge_graph.db    # SQLite query database
```
