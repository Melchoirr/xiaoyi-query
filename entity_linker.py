"""Entity extraction + Wikidata linking for Polymarket event titles."""

import json
import time
import re
import requests
from datetime import datetime
from config import (
    WIKIDATA_SEARCH_URL, WIKIDATA_SPARQL_URL, WIKIDATA_RATE_LIMIT,
    ENTITIES_FILE, ENTITY_TYPE_MAP
)

# Load spaCy once
import spacy
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ---------------------------------------------------------------------------
# Step 1: Entity extraction from event title text
# ---------------------------------------------------------------------------

def extract_entities_from_title(title):
    """Run spaCy NER on a title, return list of {text, label, type} dicts."""
    nlp = _get_nlp()
    doc = nlp(title)
    entities = []
    seen = set()

    for ent in doc.ents:
        text = ent.text.strip()
        label = ent.label_
        # Normalize labels we care about
        if label in ENTITY_TYPE_MAP and text.lower() not in seen:
            entities.append({
                "text": text,
                "spacy_label": label,
                "type": ENTITY_TYPE_MAP[label],
            })
            seen.add(text.lower())

    # Post-process: fix spaCy misclassifications
    KNOWN_TYPES = {
        "bitcoin": "cryptocurrency",
        "ethereum": "cryptocurrency",
        "polymarket": "platform",
        "gdelt": "platform",
        "nato": "organization",
        "eu": "organization",
        "g7": "organization",
        "g20": "organization",
        "sec": "organization",
        "fed": "organization",
        "ai": "technology",
    }
    for ent in entities:
        if ent["text"].lower() in KNOWN_TYPES:
            ent["type"] = KNOWN_TYPES[ent["text"].lower()]

    # Fallback: extract capitalized multi-word phrases as candidate entities
    if not entities:
        candidates = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', title)
        for c in candidates:
            if len(c) > 3 and c.lower() not in {'the', 'this', 'that', 'what', 'will'}:
                entities.append({
                    "text": c,
                    "spacy_label": "FALLBACK",
                    "type": "unknown",
                })

    return entities


# ---------------------------------------------------------------------------
# Step 2: Wikidata entity search
# ---------------------------------------------------------------------------

def search_wikidata(entity_text, entity_type=None, limit=5, context_words=None):
    """Search Wikidata for an entity, return candidate Q-ID list with scores.

    If the top results are 'family name' / 'surname' and we expect a person,
    falls back to Wikipedia search + Wikidata ID lookup.
    """
    time.sleep(WIKIDATA_RATE_LIMIT)

    def _do_wikidata_search(search_text):
        params = {
            "action": "wbsearchentities",
            "search": search_text,
            "language": "en",
            "format": "json",
            "limit": limit,
            "type": "item",
        }
        headers = {"User-Agent": "KnowledgeGraphBot/1.0 (research project)"}
        resp = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return []
        return resp.json().get("search", [])

    def _wikipedia_search(query):
        """Use Wikipedia full-text search to find relevant pages."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
        }
        headers = {"User-Agent": "KnowledgeGraphBot/1.0 (research project)"}
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params=params, headers=headers, timeout=15
        )
        if resp.status_code != 200:
            return []
        return [r["title"] for r in resp.json().get("query", {}).get("search", [])]

    def _wikipedia_title_to_qid(title):
        """Convert Wikipedia page title to Wikidata Q-ID."""
        params = {
            "action": "wbgetentities",
            "sites": "enwiki",
            "titles": title,
            "format": "json",
        }
        headers = {"User-Agent": "KnowledgeGraphBot/1.0 (research project)"}
        resp = requests.get(WIKIDATA_SEARCH_URL, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            return None
        entities = resp.json().get("entities", {})
        for qid, data in entities.items():
            if qid.startswith("Q"):
                return {
                    "qid": qid,
                    "label": data.get("labels", {}).get("en", {}).get("value", title),
                    "description": data.get("descriptions", {}).get("en", {}).get("value", ""),
                    "score": 0,
                    "url": f"//www.wikidata.org/wiki/{qid}",
                }
        return None

    results = _do_wikidata_search(entity_text)

    # Detect surname-only match when we expect a person
    is_surname_match = False
    if results and entity_type == "person":
        top_desc = (results[0].get("description") or "").lower()
        if any(w in top_desc for w in ("family name", "surname", "last name")):
            is_surname_match = True

    # Fallback: Wikipedia search for the actual person
    if is_surname_match and context_words:
        significant = [w for w in context_words
                       if len(w) > 3 and w.lower() != entity_text.lower()
                       and w.lower() not in {"will", "that", "this", "what", "which", "there",
                                              "sell", "sells", "any", "before", "after"}]
        for ctx_word in significant[:4]:
            query = f"{entity_text} {ctx_word}"
            wp_titles = _wikipedia_search(query)
            for title in wp_titles[:3]:
                wd_result = _wikipedia_title_to_qid(title)
                if wd_result:
                    # Verify it's actually a person (description contains person indicators)
                    desc = (wd_result["description"] or "").lower()
                    if any(t in desc for t in (
                        "politician", "president", "minister", "leader", "senator",
                        "governor", "mayor", "ceo", "founder", "chairman", "director",
                        "actor", "singer", "athlete", "player", "american", "british",
                        "french", "german", "italian", "spanish", "canadian",
                        "business executive", "entrepreneur", "economist", "writer",
                    )):
                        results = [wd_result] + results
                        break
            if len(results) > sum(1 for r in results if
                                   any(w in (r.get("description","")).lower()
                                       for w in ("family name", "surname"))):
                break  # found a person result, stop trying

    normalized = []
    for r in results:
        normalized.append({
            "qid": r.get("qid", r.get("id", "")),
            "label": r.get("label", ""),
            "description": r.get("description", ""),
            "score": r.get("score", 0),
            "url": r.get("url", ""),
        })
    return normalized


# ---------------------------------------------------------------------------
# Step 3: Entity disambiguation using event context
# ---------------------------------------------------------------------------

def disambiguate_entity(candidates, entity_type, event_context):
    """Pick the best Wikidata match using event category, description, and tags.

    Strategy:
    - Heavy bonus for exact label match (Wikidata label == search text)
    - Trust Wikidata API ranking (first results are usually best)
    - Use context overlap as tiebreaker only
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    context_words = set(event_context.lower().split())

    def score(c, idx):
        s = 0
        # Wikidata API position boost (first = best)
        s += max(0, (5 - idx)) * 2
        desc_lower = (c["description"] or "").lower()
        label_lower = (c["label"] or "").lower()
        # Overlap with event context (weak signal)
        desc_tokens = set(desc_lower.split())
        s += len(desc_tokens & context_words) * 2
        # Bonus when entity type appears in description
        if entity_type and entity_type.lower() in desc_lower:
            s += 2
        return s

    ranked = sorted(enumerate(candidates), key=lambda x: score(x[1], x[0]), reverse=True)
    return ranked[0][1]


# ---------------------------------------------------------------------------
# Step 4: Wikidata relation expansion (1-2 hop)
# ---------------------------------------------------------------------------

def expand_relations(qid, max_results=20):
    """SPARQL expansion: direct relations + industry-based competitors + products.

    Three strategies merged:
    1. Direct outgoing edges on a whitelist of news-relevant properties
    2. Inverse product lookup: ?item wdt:P178 (developer) this entity → products
    3. Industry-shared competitors: same P452 industry → peer companies
    """
    time.sleep(WIKIDATA_RATE_LIMIT * 2)

    headers = {"Accept": "application/json", "User-Agent": "KnowledgeGraphBot/1.0 (research project)"}
    results = []
    seen_targets = set()

    def run_sparql(query):
        try:
            resp = requests.get(WIKIDATA_SPARQL_URL, params={"query": query},
                                headers=headers, timeout=30)
            if resp.status_code != 200:
                return []
            return resp.json().get("results", {}).get("bindings", [])
        except Exception as e:
            print(f"  [SPARQL] Exception: {e}")
            return []

    def parse_bindings(bindings):
        out = []
        for b in bindings:
            target_uri = b.get("target", {}).get("value", "")
            target_qid = target_uri.split("/")[-1] if "/" in target_uri else target_uri
            if target_qid in seen_targets:
                continue
            seen_targets.add(target_qid)
            out.append({
                "relation": b.get("relation", {}).get("value", ""),
                "relation_label": b.get("relationLabel", {}).get("value", ""),
                "target": target_qid,
                "target_label": b.get("targetLabel", {}).get("value", ""),
                "target_description": b.get("targetDescription", {}).get("value", ""),
            })
        return out

    # --- Query 1: Direct relations (outgoing edges via whitelist) ---
    RELEVANT_PROPS = [
        "wdt:P108", "wdt:P1416", "wdt:P101", "wdt:P106", "wdt:P17", "wdt:P131",
        "wdt:P937", "wdt:P159", "wdt:P749", "wdt:P355", "wdt:P463", "wdt:P361",
        "wdt:P276", "wdt:P710", "wdt:P1344", "wdt:P921", "wdt:P452", "wdt:P1056",
        "wdt:P127", "wdt:P1830", "wdt:P112", "wdt:P169", "wdt:P488", "wdt:P1308",
        "wdt:P39", "wdt:P102", "wdt:P3373", "wdt:P26", "wdt:P40",
        "wdt:P178",   # developer (for products: GPU, CUDA, etc.)
    ]
    prop_filter = " || ".join(f"?p = {p}" for p in RELEVANT_PROPS)

    q1 = """
    SELECT DISTINCT ?relation ?relationLabel ?target ?targetLabel ?targetDescription WHERE {
      wd:%s ?p ?target .
      FILTER(%s)
      ?relation wikibase:directClaim ?p .
      OPTIONAL { ?target wdt:P31 ?instanceOf . }
      FILTER(
        !BOUND(?instanceOf) ||
        ?instanceOf NOT IN (wd:Q253623, wd:Q13442814, wd:Q7318358,
                            wd:Q101352, wd:Q827335, wd:Q4167410)
      )
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT %d
    """ % (qid, prop_filter, max_results)
    results.extend(parse_bindings(run_sparql(q1)))

    # --- Query 2: Inverse developer lookup (products) ---
    q2 = """
    SELECT ?target ?targetLabel ?targetDescription WHERE {
      ?target wdt:P178 wd:%s .
      ?target wikibase:sitelinks ?sitelinks .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY DESC(?sitelinks)
    LIMIT 20
    """ % qid
    for b in run_sparql(q2):
        target_uri = b.get("target", {}).get("value", "")
        target_qid = target_uri.split("/")[-1] if "/" in target_uri else target_uri
        if target_qid in seen_targets:
            continue
        seen_targets.add(target_qid)
        results.append({
            "relation": "http://www.wikidata.org/entity/P178",
            "relation_label": "developer of (product)",
            "target": target_qid,
            "target_label": b.get("targetLabel", {}).get("value", ""),
            "target_description": b.get("targetDescription", {}).get("value", ""),
        })

    # --- Query 3: Industry-shared competitors (ranked by notability) ---
    q3 = """
    SELECT DISTINCT ?target ?targetLabel ?targetDescription WHERE {
      wd:%s wdt:P452 ?industry .
      ?target wdt:P452 ?industry .
      FILTER(?target != wd:%s)
      ?target wikibase:sitelinks ?sitelinks .
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    ORDER BY DESC(?sitelinks)
    LIMIT 10
    """ % (qid, qid)
    for b in run_sparql(q3):
        target_uri = b.get("target", {}).get("value", "")
        target_qid = target_uri.split("/")[-1] if "/" in target_uri else target_uri
        if target_qid in seen_targets:
            continue
        seen_targets.add(target_qid)
        results.append({
            "relation": "http://www.wikidata.org/entity/P452",
            "relation_label": "competitor (shared industry)",
            "target": target_qid,
            "target_label": b.get("targetLabel", {}).get("value", ""),
            "target_description": b.get("targetDescription", {}).get("value", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Step 5: Full entity linking pipeline
# ---------------------------------------------------------------------------

def run_entity_linking(events_data, expand=True):
    """Process all events: extract entities, link to Wikidata, optionally expand.

    events_data: list of event dicts from polymarket pipeline.

    Returns list of entity records saved to ENTITIES_FILE.
    """
    nlp = _get_nlp()
    entity_map = {}  # normalized text -> wikidata info
    results = []

    for event in events_data:
        title = event["title"]
        description = event.get("description", "")
        category = event.get("category", "")
        tags_str = " ".join(event.get("tags", []))
        event_context = f"{description} {category} {tags_str}"

        raw_entities = extract_entities_from_title(title)
        event_entities = []

        for ent in raw_entities:
            text = ent["text"]
            etype = ent["type"]

            # Check cache
            key = text.lower()
            if key in entity_map:
                event_entities.append(entity_map[key])
                continue

            # Search Wikidata
            print(f"  [NER] {text} ({etype})")
            context_tokens = event_context.lower().split()
            candidates = search_wikidata(text, etype, context_words=context_tokens)

            # Disambiguate
            best = disambiguate_entity(candidates, etype, event_context)

            # Expand relations
            related = []
            if best and expand:
                print(f"    -> {best['qid']} ({best['label']}): {best['description']}")
                related = expand_relations(best["qid"])
                print(f"    -> {len(related)} related entities")

            entity_record = {
                "text": text,
                "spacy_label": ent["spacy_label"],
                "type": etype,
                "candidates": candidates[:5],
                "best_match": best,
                "related_entities": related,
            }
            entity_map[key] = entity_record
            event_entities.append(entity_record)

        results.append({
            "event_id": event["event_id"],
            "title": title,
            "entities": event_entities,
        })

    # Save
    with open(ENTITIES_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[EntityLinker] Saved entities for {len(results)} events to {ENTITIES_FILE}")

    return results


if __name__ == "__main__":
    from polymarket_client import run_polymarket_pipeline
    events, _ = run_polymarket_pipeline(event_limit=10)
    run_entity_linking(events)
