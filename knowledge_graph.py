"""Knowledge graph builder: unifies Polymarket entities into a graph structure
and provides a GDELT news matching interface (mock for now, real when data arrives).
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from config import KNOWLEDGE_GRAPH_FILE, DATA_DIR


# ---------------------------------------------------------------------------
# Graph builder from entity linking results
# ---------------------------------------------------------------------------

def build_graph(entity_results, events_data, curves_data):
    """Build a unified knowledge graph from entity linking results.

    Graph structure:
      nodes: {qid -> {label, type, description, properties}}
      edges: [{source, target, relation, event_ids}]
      event_index: {qid -> [event_id, ...]}
    """
    nodes = {}
    edges = []
    event_entity_index = {}  # event_id -> [qid, ...]

    for evt_result in entity_results:
        event_id = evt_result["event_id"]
        event_entity_index[event_id] = []

        for ent in evt_result["entities"]:
            best = ent.get("best_match")
            if not best:
                continue

            qid = best["qid"]
            event_entity_index[event_id].append(qid)

            # Add node
            if qid not in nodes:
                nodes[qid] = {
                    "qid": qid,
                    "label": best["label"],
                    "type": ent["type"],
                    "description": best["description"],
                    "source_text": ent["text"],
                    "events": set(),
                }
            nodes[qid]["events"].add(event_id)

            # Add edges from relation expansion
            for rel in ent.get("related_entities", []):
                target_qid = rel["target"].split("/")[-1] if "wikidata" in rel["target"] else rel["target"]
                if len(target_qid) < 3 or not target_qid.startswith("Q"):
                    continue

                edges.append({
                    "source": qid,
                    "target": target_qid,
                    "relation": rel.get("relation", ""),
                    "relation_label": rel.get("relation_label", ""),
                    "event_ids": [event_id],
                })

                # Add target node if not exists
                if target_qid not in nodes:
                    nodes[target_qid] = {
                        "qid": target_qid,
                        "label": rel.get("target_label", target_qid),
                        "type": "related",
                        "description": rel.get("target_description", ""),
                        "source_text": "",
                        "events": set(),
                    }

    # Convert sets to lists for JSON serialization
    for node in nodes.values():
        node["events"] = list(node["events"])

    graph = {
        "built_at": datetime.now().isoformat(),
        "nodes": list(nodes.values()),
        "edges": edges,
        "event_entity_index": {eid: list(set(qids))
                               for eid, qids in event_entity_index.items()},
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "total_events": len(event_entity_index),
        },
    }

    with open(KNOWLEDGE_GRAPH_FILE, "w") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
    print(f"[KnowledgeGraph] Saved graph: {graph['stats']}")

    return graph


# ---------------------------------------------------------------------------
# GDELT news matching interface (mock)
# ---------------------------------------------------------------------------

class GDELTMatcher:
    """Interface for matching GDELT news articles to knowledge graph entities.

    Currently uses mock data. When GDELT GKG data is available, implement:
      - load_gkg(path): load GKG CSV/Pandas dataframe
      - match_by_entity(qids, time_window): filter GKG rows where entities match
      - score_article(article, event): compute relevance score
    """

    def __init__(self, graph):
        self.graph = graph
        self.node_index = {n["qid"]: n for n in graph["nodes"]}

    def query_entities(self, qids):
        """Get entity details by Q-ID list."""
        return [self.node_index[q] for q in qids if q in self.node_index]

    def get_entity_qids_for_event(self, event_id):
        """Return all Q-IDs linked to a given event."""
        return self.graph["event_entity_index"].get(event_id, [])

    def find_related_events(self, qid):
        """Find events that share entities with the given Q-ID."""
        related = set()
        for edge in self.graph["edges"]:
            if edge["source"] == qid or edge["target"] == qid:
                related.update(edge["event_ids"])
        return list(related)

    # --- Mock GDELT matching ---
    def match_news(self, event_id, time_window_days=7):
        """Mock: return placeholder news matches for an event.

        When real GDELT data arrives, this will:
        1. Get entity Q-IDs for the event
        2. Query GDELT GKG for articles mentioning those entities
        3. Score and rank by relevance
        4. Return top matches with article metadata and scores
        """
        qids = self.get_entity_qids_for_event(event_id)
        entities = self.query_entities(qids)

        return {
            "event_id": event_id,
            "matched_entities": entities,
            "mock_articles": [
                {
                    "title": f"[MOCK] News about {e['label']}",
                    "url": f"https://example.com/news/{e['qid']}",
                    "date": datetime.now().isoformat(),
                    "entities_matched": [e["qid"]],
                    "relevance_score": 1.0,
                    "source": "MOCK_GDELT",
                }
                for e in entities
            ],
            "note": "Replace with real GDELT GKG query when data is available. "
                    "Filter GKG by V2Locations, V2Persons, V2Organizations columns "
                    "which contain Wikidata IDs.",
        }

    # --- GDELT GKG loader (stub) ---
    @staticmethod
    def load_gkg(filepath):
        """Stub: load a GDELT GKG file and return records matching entity Q-IDs.

        GKG V2 format (version 2.0+) columns of interest:
          - col 5: V2Persons (comma-separated Wikidata IDs)
          - col 6: V2Organizations (comma-separated Wikidata IDs)
          - col 7: V2Locations (comma-separated Wikidata IDs)
          - col 3: V2Themes (comma-separated theme IDs)
          - col 10: DocumentIdentifier (URL to source article)

        Args:
            filepath: Path to GKG CSV/TSV file

        Returns:
            List of article dicts with extracted entity IDs
        """
        raise NotImplementedError(
            "GDELT GKG data not available on this device. "
            "Place your GKG file and call this method."
        )


# ---------------------------------------------------------------------------
# SQLite-backed knowledge graph (for query performance)
# ---------------------------------------------------------------------------

class KnowledgeGraphDB:
    """SQLite-backed knowledge graph for efficient queries."""

    def __init__(self, db_path=None):
        self.db_path = db_path or str(DATA_DIR / "knowledge_graph.db")
        self.conn = sqlite3.connect(self.db_path)
        self._create_tables()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                qid TEXT PRIMARY KEY,
                label TEXT,
                type TEXT,
                description TEXT,
                source_text TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                target TEXT,
                relation TEXT,
                relation_label TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS event_entities (
                event_id TEXT,
                entity_qid TEXT,
                PRIMARY KEY (event_id, entity_qid)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                title TEXT,
                category TEXT
            )
        """)
        self.conn.commit()

    def load_from_graph(self, graph):
        """Populate SQLite from JSON graph."""
        cur = self.conn.cursor()
        for node in graph["nodes"]:
            cur.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?, ?, ?, ?, ?)",
                (node["qid"], node["label"], node["type"],
                 node.get("description", ""), node.get("source_text", ""))
            )
            for evt_id in node.get("events", []):
                cur.execute(
                    "INSERT OR IGNORE INTO event_entities VALUES (?, ?)",
                    (evt_id, node["qid"])
                )

        for edge in graph["edges"]:
            cur.execute(
                "INSERT INTO edges (source, target, relation, relation_label) VALUES (?, ?, ?, ?)",
                (edge["source"], edge["target"],
                 edge.get("relation", ""), edge.get("relation_label", ""))
            )

        for eid, qids in graph["event_entity_index"].items():
            evt = next((e for e in graph.get("_events_raw", []) if e["event_id"] == eid), None)
            title = evt["title"] if evt else eid
            cur.execute(
                "INSERT OR IGNORE INTO events VALUES (?, ?, ?)",
                (eid, title, "")
            )

        self.conn.commit()

    def get_entities_for_event(self, event_id):
        return self.conn.execute(
            "SELECT n.* FROM nodes n "
            "JOIN event_entities ee ON n.qid = ee.entity_qid "
            "WHERE ee.event_id = ?", (event_id,)
        ).fetchall()

    def get_coreachable_entities(self, qid, max_hops=2):
        """Find all entities reachable from qid within max_hops."""
        seen = {qid}
        frontier = {qid}
        for _ in range(max_hops):
            next_frontier = set()
            for f in frontier:
                rows = self.conn.execute(
                    "SELECT source, target FROM edges WHERE source = ? OR target = ?",
                    (f, f)
                ).fetchall()
                for src, tgt in rows:
                    if src not in seen:
                        seen.add(src)
                        next_frontier.add(src)
                    if tgt not in seen:
                        seen.add(tgt)
                        next_frontier.add(tgt)
            frontier = next_frontier
        return list(seen)

    def search_entity(self, keyword):
        """Full-text-like search on entity labels."""
        return self.conn.execute(
            "SELECT * FROM nodes WHERE label LIKE ? OR description LIKE ?",
            (f"%{keyword}%", f"%{keyword}%")
        ).fetchall()


if __name__ == "__main__":
    # Test with saved data
    with open(KNOWLEDGE_GRAPH_FILE) as f:
        graph = json.load(f)
    db = KnowledgeGraphDB()
    db.load_from_graph(graph)
    print(f"Loaded {len(db.conn.execute('SELECT * FROM nodes').fetchall())} nodes into DB")

    matcher = GDELTMatcher(graph)
    # Demo: query first event
    first_event = list(graph["event_entity_index"].keys())[0]
    result = matcher.match_news(first_event)
    print(f"\nDemo match_news for {first_event}:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
