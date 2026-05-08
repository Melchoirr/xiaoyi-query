"""LLM-based entity relationship enricher — supplements Wikidata gaps.

Uses DeepSeek API to infer relationships Wikidata doesn't model:
- B2B customer relationships (e.g., NVIDIA → Microsoft, Google)
- Strategic partnerships
- Competitive landscape context
- Supply chain / industry dynamics

Design principles for anti-hallucination:
- Structured JSON output with mandatory confidence field
- Wikidata edges take precedence; LLM only fills gaps
- Cached results avoid duplicate API calls
- No API key = graceful fallback to empty results
"""

import json
import os
import time
from pathlib import Path
from config import DATA_DIR


# ---------------------------------------------------------------------------
# Prompt design
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a knowledge graph assistant. Your job is to supplement missing
business relationships for entities, outputting ONLY valid JSON.

For a given entity (company, person, technology, country), list:
1. key_customers: organizations that purchase/use this entity's products or services
2. key_competitors: organizations competing in the same market
3. key_partners: strategic partners, alliances, joint ventures
4. key_technologies: associated technologies, platforms, or tools

Rules:
- Only include major, well-known relationships
- Each entry must have: name (string), relationship (short description), confidence (high/medium/low)
- "high" = publicly confirmed (SEC filing, press release, official partnership)
- "medium" = widely reported/analyzed but no official confirmation
- "low" = industry speculation
- Output ONLY the JSON array. No explanation, no markdown."""

FEW_SHOT_EXAMPLE = """
Example for "NVIDIA" (entity_type=organization):
[
  {"category": "key_customers", "name": "Microsoft", "relationship": "purchases GPUs for Azure AI infrastructure", "confidence": "high"},
  {"category": "key_customers", "name": "Meta", "relationship": "purchases GPUs for AI training clusters", "confidence": "high"},
  {"category": "key_customers", "name": "Google Cloud", "relationship": "purchases GPUs for cloud AI offerings", "confidence": "high"},
  {"category": "key_customers", "name": "OpenAI", "relationship": "uses NVIDIA GPUs for GPT model training", "confidence": "high"},
  {"category": "key_customers", "name": "Tesla", "relationship": "uses NVIDIA GPUs for autonomous driving training", "confidence": "medium"},
  {"category": "key_competitors", "name": "AMD", "relationship": "competes in GPU and datacenter accelerator market", "confidence": "high"},
  {"category": "key_competitors", "name": "Intel", "relationship": "competes in datacenter AI accelerators", "confidence": "high"},
  {"category": "key_partners", "name": "TSMC", "relationship": "manufactures NVIDIA's GPU chips", "confidence": "high"},
  {"category": "key_partners", "name": "Dell Technologies", "relationship": "OEM partner for enterprise GPU servers", "confidence": "high"},
  {"category": "key_technologies", "name": "CUDA", "relationship": "NVIDIA's parallel computing platform", "confidence": "high"},
  {"category": "key_technologies", "name": "DLSS", "relationship": "deep learning super sampling technology", "confidence": "high"}
]
"""


def build_user_prompt(entity_name, entity_type, entity_description="", wiki_relations=None):
    """Build a structured user prompt for the LLM."""
    wiki_context = ""
    if wiki_relations:
        existing = [r["target_label"] for r in wiki_relations[:10]
                    if r.get("target_label")]
        if existing:
            wiki_context = f"\nAlready known relationships (from Wikidata, do NOT repeat): {', '.join(existing)}"

    return f"""Entity: {entity_name}
Type: {entity_type}
Description: {entity_description}{wiki_context}

List the key customers, competitors, partners, and technologies for this entity.
Output ONLY valid JSON following the exact format from the example."""


# ---------------------------------------------------------------------------
# DeepSeek API client
# ---------------------------------------------------------------------------

class LLMEnricher:
    """Enriches entities with LLM-inferred relationships.

    Usage:
        enricher = LLMEnricher(api_key="...")   # real mode
        enricher = LLMEnricher()                 # dry-run / no-key mode
    """

    def __init__(self, api_key=None, base_url="https://api.deepseek.com",
                 model="deepseek-chat", cache_file=None):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model
        self.cache_file = cache_file or str(DATA_DIR / "llm_relations_cache.json")
        self._cache = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file) as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self):
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    @property
    def is_available(self):
        return bool(self.api_key)

    def enrich(self, entity_name, entity_type, entity_description="",
               wiki_relations=None):
        """Return LLM-inferred relations for an entity. Returns empty list if no key.

        Returns list of: {category, name, relationship, confidence, source: "llm"}
        """
        cache_key = f"{entity_name}|{entity_type}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        if not self.is_available:
            print(f"  [LLM] No API key configured, skipping {entity_name}")
            return []

        user_prompt = build_user_prompt(entity_name, entity_type,
                                        entity_description, wiki_relations)

        result = self._call_deepseek(user_prompt)
        self._cache[cache_key] = result
        self._save_cache()
        return result

    def _call_deepseek(self, user_prompt, max_retries=2):
        """Call DeepSeek chat API, parse JSON response."""
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": FEW_SHOT_EXAMPLE},
                {"role": "assistant", "content": "I understand. I will output only valid JSON following this format."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 1024,
        }

        for attempt in range(max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=headers, json=payload, timeout=30
                )
                if resp.status_code != 200:
                    print(f"  [LLM] API error {resp.status_code}: {resp.text[:200]}")
                    return []

                content = resp.json()["choices"][0]["message"]["content"]
                # Strip markdown code fences if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                relations = json.loads(content)
                if not isinstance(relations, list):
                    return []

                # Validate and tag
                valid = []
                for r in relations:
                    if all(k in r for k in ("category", "name", "relationship", "confidence")):
                        r["source"] = "llm"
                        valid.append(r)
                return valid

            except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    print(f"  [LLM] Failed to parse response: {e}")
                    return []

        return []

    def enrich_batch(self, entities, wiki_relations_map=None):
        """Enrich multiple entities. Returns {entity_name: [relations]}."""
        results = {}
        for ent_name, ent_type, ent_desc in entities:
            wiki_rels = None
            if wiki_relations_map:
                wiki_rels = wiki_relations_map.get(ent_name)
            results[ent_name] = self.enrich(ent_name, ent_type, ent_desc, wiki_rels)
        return results


# ---------------------------------------------------------------------------
# Integration with knowledge graph
# ---------------------------------------------------------------------------

def merge_llm_relations(graph_data, enricher):
    """Merge LLM-inferred relations into knowledge graph JSON.

    For each entity node in the graph, query LLM for missing relations.
    Returns updated graph with new nodes and edges tagged source="llm".
    """
    if not enricher.is_available:
        print("[LLM] Skipping enrichment — no API key")
        return graph_data

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    new_nodes = []
    new_edges = []
    existing_qids = {n["qid"] for n in nodes}

    for node in nodes:
        if node.get("type") != "organization":
            continue

        rels = enricher.enrich(
            node["label"], node.get("type", "organization"),
            node.get("description", ""),
            wiki_relations=[e for e in edges if e["source"] == node["qid"]]
        )

        for r in rels:
            # Create a synthetic ID for entities not in Wikidata
            name_key = r["name"].lower().replace(" ", "_")
            synthetic_qid = f"llm:{name_key}"

            if synthetic_qid not in existing_qids:
                existing_qids.add(synthetic_qid)
                new_nodes.append({
                    "qid": synthetic_qid,
                    "label": r["name"],
                    "type": r["category"],
                    "description": r["relationship"],
                    "source_text": "",
                    "events": list(node.get("events", [])),
                    "source": "llm",
                })

            new_edges.append({
                "source": node["qid"],
                "target": synthetic_qid,
                "relation": "llm_inferred",
                "relation_label": r["category"],
                "event_ids": list(node.get("events", [])),
                "confidence": r["confidence"],
                "source": "llm",
            })

    print(f"[LLM] Added {len(new_nodes)} nodes, {len(new_edges)} edges")
    graph_data["nodes"].extend(new_nodes)
    graph_data["edges"].extend(new_edges)
    graph_data["stats"]["total_nodes"] += len(new_nodes)
    graph_data["stats"]["total_edges"] += len(new_edges)
    return graph_data
