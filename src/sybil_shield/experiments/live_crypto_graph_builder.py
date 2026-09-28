import json
import os
import random
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib import parse, request


def _jsonrpc_call(method: str, params: Sequence[Any], url: str = "https://eth.llamarpc.com") -> Any:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)}).encode("utf-8")
    req = request.Request(url, data=payload, headers={"Content-Type": "application/json", "User-Agent": "sybil-shield/0.1"})
    with request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if "error" in data and data["error"]:
        raise RuntimeError(f"JSON-RPC {method} failed: {data['error']}")
    return data.get("result")

import networkx as nx

from sybil_shield.core.graph_store import save_graph


DEFAULT_PROVIDER = "etherscan"


def _http_get_json(url: str, timeout: int = 30) -> Any:
    req = request.Request(url, headers={"User-Agent": "sybil-shield/0.1"})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _normalize_address(addr: str) -> str:
    return str(addr).lower()


def _api_key() -> str:
    key = os.environ.get("ETHERSCAN_API_KEY") or os.environ.get("ALCHEMY_API_KEY")
    if not key:
        raise RuntimeError(
            "No crypto API key configured. Set ETHERSCAN_API_KEY or ALCHEMY_API_KEY before running the live graph builder."
        )
    return key


def _as_hex_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    v = value.strip()
    if v.startswith("0x") or v.startswith("0X"):
        try:
            return int(v, 16)
        except ValueError:
            return None
    return None


def fetch_etherscan_transactions(address: str, page_size: int = 1000, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch the latest transactions for one wallet using Etherscan's account API."""
    address = _normalize_address(address)
    api_key = api_key or _api_key()
    params = {
        "module": "account",
        "action": "txlist",
        "address": address,
        "startblock": 0,
        "endblock": 99999999,
        "page": 1,
        "offset": page_size,
        "sort": "asc",
        "apikey": api_key,
    }
    for base_url in (
        "https://api.etherscan.io/v2/api?chainid=1",
        "https://api.etherscan.io/api?",
    ):
        url = base_url + parse.urlencode(params) if base_url.endswith("?") else base_url + "&" + parse.urlencode(params)
        try:
            payload = _http_get_json(url)
        except Exception:
            continue
        if payload.get("status") == "1" or payload.get("result") == []:
            return payload.get("result") or []
        msg = payload.get("message") or "unknown etherscan response"
        if "No transactions found" in msg or payload.get("result") == []:
            return []
        if "deprecated" in str(msg).lower():
            continue
        raise RuntimeError(f"Etherscan API error for {address}: {msg}")
    raise RuntimeError(f"Etherscan API error for {address}: request failed unexpectedly")


def _fetch_random_active_wallet(api_key: Optional[str] = None, max_attempts: int = 5) -> str:
    """Pick a real, recently-active wallet by grabbing a random recent
    block and taking one of its transaction participants.

    A literally-random 160-bit address would almost certainly never
    have been used on-chain, so it's a useless BFS seed (zero
    transactions to expand from). Sampling from an actual recent
    block's transactions guarantees the seed has at least one real,
    confirmed transaction to start the crawl from.
    """
    api_key = api_key or _api_key()

    latest_block = None
    for base_url in (
        "https://api.etherscan.io/v2/api?chainid=1",
        "https://api.etherscan.io/api?",
    ):
        url = base_url + parse.urlencode({"module": "proxy", "action": "eth_blockNumber", "apikey": api_key}) if base_url.endswith("?") else base_url + "&" + parse.urlencode({"module": "proxy", "action": "eth_blockNumber", "apikey": api_key})
        try:
            latest = _http_get_json(url)
        except Exception:
            latest = None
        latest_block = _as_hex_int((latest or {}).get("result")) if isinstance(latest, dict) else None
        if latest_block is not None:
            break
        msg = (latest or {}).get("message") if isinstance(latest, dict) else ""
        if "deprecated" in str(msg).lower():
            continue
    if latest_block is None:
        try:
            latest_block = _as_hex_int(_jsonrpc_call("eth_blockNumber", []))
        except Exception:
            latest_block = None

    if latest_block is None:
        # Last-resort fallback for transient Etherscan deprecations or
        # network outages: sample from a recent block window instead of
        # crashing. Ethereum mainnet is millions of blocks deep, so a
        # block height in this rough neighborhood is still a valid seed
        # source for a live-wallet crawl.
        latest_block = 20_000_000

    for _ in range(max_attempts):
        # Stay a little behind the tip (fully indexed) and look back
        # over a wide recent window for variety across runs.
        block_number = random.randint(max(0, latest_block - 50_000), max(0, latest_block - 10))
        block = None
        for base_url in (
            "https://api.etherscan.io/v2/api?chainid=1",
            "https://api.etherscan.io/api?",
        ):
            try:
                url = base_url + parse.urlencode({
                    "module": "proxy",
                    "action": "eth_getBlockByNumber",
                    "tag": hex(block_number),
                    "boolean": "true",
                    "apikey": api_key,
                }) if base_url.endswith("?") else base_url + "&" + parse.urlencode({
                    "module": "proxy",
                    "action": "eth_getBlockByNumber",
                    "tag": hex(block_number),
                    "boolean": "true",
                    "apikey": api_key,
                })
                block = _http_get_json(url)
                if block.get("message") and "deprecated" in str(block.get("message")).lower():
                    block = None
                    continue
                break
            except Exception:
                block = None
        if block is None:
            try:
                raw = _jsonrpc_call("eth_getBlockByNumber", [hex(block_number), True])
                txs = (raw or {}).get("transactions") or []
            except Exception:
                txs = []
        else:
            txs = (block.get("result") or {}).get("transactions") or []
        if not txs:
            continue
        tx = random.choice(txs)
        addr = tx.get("from") or tx.get("to")
        if addr:
            return _normalize_address(addr)

    raise RuntimeError(
        f"Could not find an active wallet after {max_attempts} random blocks -- "
        "try again (transient: an unlucky run of empty blocks)."
    )


def build_live_wallet_graph(
    seed_wallet: Optional[str] = None,
    max_nodes: int = 200,
    max_depth: int = 2,
    page_size: int = 1000,
    api_key: Optional[str] = None,
    graph_type: str = "ethereum_wallet",
    persist: bool = False,
    root_dir: Optional[str] = None,
    provider: str = DEFAULT_PROVIDER,
    event_cb: Optional[Any] = None,
) -> nx.Graph:
    """Build a real wallet graph by expanding from a seed wallet through ETH transactions.

    The graph is intentionally conservative: it expands breadth-first, deduplicates addresses,
    and loads a bounded number of transactions per wallet before storing the snapshot.

    seed_wallet: omit (or pass None) to start from a random, recently-active
    wallet instead of a specific address -- see _fetch_random_active_wallet.

    event_cb: optional callable(dict) invoked as the crawl progresses --
    {"type": "node", "id": ..., "node_type": "wallet"} when a wallet is
    first added, {"type": "edge", "source": ..., "target": ...} when an
    edge is first added, and {"type": "progress", "wallets_found": ...,
    "edges_found": ..., "api_calls_used": ...} after each wallet's
    txlist is processed. Matches the visualizer's ScanEvent shape
    (sybil-types.ts) so the API layer can forward these straight over a
    WebSocket without reshaping them. Never called with a "done" event --
    that's the API layer's job, once scoring (not just crawling) finishes.
    """
    def _emit(evt: Dict[str, Any]) -> None:
        if event_cb is not None:
            event_cb(evt)
    api_key = api_key or _api_key()
    if seed_wallet is None:
        seed_wallet = _fetch_random_active_wallet(api_key=api_key)
        print(f"No --seed-wallet given; picked random active wallet: {seed_wallet}")
    seed_wallet = _normalize_address(seed_wallet)
    G = nx.Graph()
    G.add_node(seed_wallet, label=0, node_type="wallet", role="seed", source="ethereum")
    _emit({"type": "node", "id": seed_wallet, "node_type": "wallet"})

    queue: Deque[Tuple[str, int]] = deque([(seed_wallet, 0)])
    seen: Set[str] = {seed_wallet}
    api_calls = 0
    # A single high-degree wallet (e.g. the seed itself, or an exchange
    # hot wallet encountered at depth 1) can otherwise have enough
    # counterparties in its txlist to burn the entire max_nodes budget
    # by itself -- G.number_of_nodes() hits max_nodes while still
    # inside that one node's expansion, the outer while exits, and
    # every deeper-depth node queued during that expansion never gets
    # dequeued. That silently truncates the crawl to depth 0 regardless
    # of max_depth, producing a pure star graph with no structural
    # signal between leaves. Capping new nodes added per popped wallet
    # keeps the budget spread across depths so max_depth is honored.
    max_new_per_node = max(1, max_nodes // max(1, max_depth * 20))
    while queue and G.number_of_nodes() < max_nodes:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        txs = fetch_etherscan_transactions(node, page_size=page_size, api_key=api_key)
        api_calls += 1
        new_this_node = 0
        for tx in txs:
            if G.number_of_nodes() >= max_nodes or new_this_node >= max_new_per_node:
                break
            tx_from = _normalize_address(tx.get("from") or "")
            tx_to = _normalize_address(tx.get("to") or "")
            if not tx_from and not tx_to:
                continue
            if tx_from and tx_from not in G:
                G.add_node(tx_from, label=0, node_type="wallet", source="ethereum")
                _emit({"type": "node", "id": tx_from, "node_type": "wallet"})
            if tx_to and tx_to not in G:
                G.add_node(tx_to, label=0, node_type="wallet", source="ethereum")
                _emit({"type": "node", "id": tx_to, "node_type": "wallet"})
            if tx_from == tx_to:
                continue
            if tx_from and tx_to:
                key = (tx_from, tx_to)
                if G.has_edge(*key):
                    G.edges[tx_from, tx_to]["tx_count"] = G.edges[tx_from, tx_to].get("tx_count", 0) + 1
                else:
                    G.add_edge(tx_from, tx_to, value_wei=float(tx.get("value") or 0), timestamp=int(tx.get("timeStamp") or 0), tx_hash=tx.get("hash"), tx_count=1)
                    _emit({"type": "edge", "source": tx_from, "target": tx_to})

            if tx_from and tx_from not in seen and tx_from != seed_wallet:
                seen.add(tx_from)
                queue.append((tx_from, depth + 1))
                new_this_node += 1
            if tx_to and tx_to not in seen and tx_to != seed_wallet:
                seen.add(tx_to)
                queue.append((tx_to, depth + 1))
                new_this_node += 1

            if G.number_of_nodes() >= max_nodes:
                break

        _emit({
            "type": "progress",
            "wallets_found": G.number_of_nodes(),
            "edges_found": G.number_of_edges(),
            "api_calls_used": api_calls,
        })

    if persist:
        meta = save_graph(
            G,
            graph_type=graph_type,
            params={
                "seed_wallet": seed_wallet,
                "max_nodes": max_nodes,
                "max_depth": max_depth,
                "page_size": page_size,
                "provider": provider,
            },
            root_dir=root_dir,
        )
        meta["api_calls_used"] = api_calls
        graph_dir = os.path.join(
            root_dir or os.environ.get("SYBIL_SHIELD_GRAPHS_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "crypto_graphs")),
            graph_type,
            meta["graph_id"],
            "meta.json",
        )
        with open(graph_dir, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return G

    return G


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch a real wallet graph from Ethereum transaction data and store it in the graph registry.")
    parser.add_argument("--seed-wallet", default=None,
                         help="Ethereum address to start from. Omit for a random, "
                              "recently-active wallet (sampled from a recent block).")
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--graph-type", default="ethereum_wallet")
    parser.add_argument("--provider", default="etherscan")
    parser.add_argument("--root-dir", default=None, help="Optional override for the graph registry root")
    parser.add_argument("--api-key", default=None, help="Optional API key override; defaults to ETHERSCAN_API_KEY or ALCHEMY_API_KEY")
    args = parser.parse_args()

    G = build_live_wallet_graph(
        seed_wallet=args.seed_wallet,
        max_nodes=args.max_nodes,
        max_depth=args.max_depth,
        page_size=args.page_size,
        api_key=args.api_key,
        graph_type=args.graph_type,
        persist=True,
        root_dir=args.root_dir,
        provider=args.provider,
    )
    print(f"Fetched graph: nodes={G.number_of_nodes()} edges={G.number_of_edges()} pattern=ethereum-wallet")


if __name__ == "__main__":
    main()