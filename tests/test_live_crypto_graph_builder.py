import json

import networkx as nx

from sybil_shield.experiments.live_crypto_graph_builder import build_live_wallet_graph


def test_build_live_wallet_graph_uses_mocked_api(monkeypatch, tmp_path):
    def fake_fetch(address, page_size=1000, api_key=None):
        if address == "0xseed":
            return [
                {"from": "0xseed", "to": "0xaaa", "value": "1000000000000000", "timeStamp": "1700000000", "hash": "tx1"},
                {"from": "0xseed", "to": "0xbbb", "value": "2000000000000000", "timeStamp": "1700000001", "hash": "tx2"},
            ]
        if address == "0xaaa":
            return [{"from": "0xaaa", "to": "0xccc", "value": "3000000000000000", "timeStamp": "1700000002", "hash": "tx3"}]
        if address == "0xbbb":
            return [{"from": "0xbbb", "to": "0xddd", "value": "4000000000000000", "timeStamp": "1700000003", "hash": "tx4"}]
        return []

    monkeypatch.setattr("sybil_shield.experiments.live_crypto_graph_builder.fetch_etherscan_transactions", fake_fetch)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "dummy-key")

    G = build_live_wallet_graph(
        seed_wallet="0xseed",
        max_nodes=10,
        max_depth=2,
        persist=True,
        root_dir=str(tmp_path),
        graph_type="ethereum_wallet",
    )

    assert G.number_of_nodes() >= 3
    assert G.number_of_edges() >= 2
    assert "0xseed" in G.nodes
    assert any(n in G.nodes for n in ["0xaaa", "0xbbb", "0xccc", "0xddd"])
    assert (tmp_path / "ethereum_wallet").exists()


def test_build_live_wallet_graph_skips_self_transfers(monkeypatch, tmp_path):
    def fake_fetch(address, page_size=1000, api_key=None):
        return [{"from": "0xseed", "to": "0xseed", "value": "123", "timeStamp": "1700000000", "hash": "self"}]

    monkeypatch.setattr("sybil_shield.experiments.live_crypto_graph_builder.fetch_etherscan_transactions", fake_fetch)
    monkeypatch.setenv("ETHERSCAN_API_KEY", "dummy-key")

    G = build_live_wallet_graph(
        seed_wallet="0xseed",
        max_nodes=10,
        max_depth=1,
        persist=False,
        graph_type="ethereum_wallet",
    )

    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0
    assert list(nx.selfloop_edges(G)) == []


def test_fetch_random_active_wallet_handles_deprecated_v1_response(monkeypatch):
    calls = []

    def fake_http_get_json(url):
        calls.append(url)
        if "eth_blockNumber" in url:
            return {
                "status": "0",
                "message": "You are using a deprecated V1endpoint, switch to Etherscan API V2 using https://docs.etherscan.io/v2-migration",
                "result": "You are using a deprecated V1endpoint, switch to Etherscan API V2 using https://docs.etherscan.io/v2-migration",
            }
        if "eth_getBlockByNumber" in url:
            return {
                "result": {
                    "transactions": [{"from": "0xaaa", "to": "0xbbb"}]
                }
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("sybil_shield.experiments.live_crypto_graph_builder._http_get_json", fake_http_get_json)

    wallet = __import__(
        "sybil_shield.experiments.live_crypto_graph_builder",
        fromlist=["_fetch_random_active_wallet"],
    )._fetch_random_active_wallet(api_key="dummy", max_attempts=2)

    assert wallet in {"0xaaa", "0xbbb"}
    assert any("eth_blockNumber" in c for c in calls)
