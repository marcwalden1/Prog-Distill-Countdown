"""Random directed graphs + shortest-path utilities for the shortest-path task.

Graph model ("k-out", chosen 2026-11-22): N nodes, each node draws its out-degree
uniformly from OUT_DEGREES and points to that many distinct uniformly-random other
nodes (no self-loops, no duplicate edges). Same mean degree (2) and in-degree
spread as a directed Erdos-Renyi G(N, 2/(N-1)), but no sink nodes and much less
rejection when conditioning on a long shortest path (77% vs 58% of graphs have a
pair at distance 6 for N=12..16). See research_logs/2026-11-22_shortest-path-task.md.

Stdlib only, so the reward function can import nothing heavier.
"""

import random
from collections import deque

OUT_DEGREES = (1, 2, 3)


def random_kout_graph(n, rng, out_degrees=OUT_DEGREES):
    """Return adjacency as {u: sorted list of successors} over nodes 0..n-1."""
    adj = {}
    for u in range(n):
        k = rng.choice(out_degrees)
        adj[u] = sorted(rng.sample([v for v in range(n) if v != u], k))
    return adj


def bfs(adj, s):
    """Shortest distance and number of shortest paths from s to every reachable node."""
    dist, cnt = {s: 0}, {s: 1}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                cnt[v] = cnt[u]
                q.append(v)
            elif dist[v] == dist[u] + 1:
                cnt[v] += cnt[u]
    return dist, cnt


def sample_shortest_path(adj, s, t, rng):
    """A uniformly random shortest s->t path (list of nodes), or None if unreachable.

    Walks backward from t choosing predecessor u with probability
    cnt[u] / cnt[v], which makes every shortest path equally likely.
    """
    dist, cnt = bfs(adj, s)
    if t not in dist:
        return None
    preds = {}
    for u, vs in adj.items():
        for v in vs:
            preds.setdefault(v, []).append(u)
    path = [t]
    v = t
    while v != s:
        cands = [u for u in preds.get(v, ()) if dist.get(u) == dist[v] - 1]
        weights = [cnt[u] for u in cands]
        v = rng.choices(cands, weights=weights, k=1)[0]
        path.append(v)
    return path[::-1]


def adj_from_edges(src, dst):
    adj = {}
    for a, b in zip(src, dst):
        adj.setdefault(int(a), []).append(int(b))
    return adj
