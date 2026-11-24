"""Tests for tasks/shortest_path. Run directly: python3 tests/test_shortest_path.py -v"""

import itertools
import os
import random
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tasks", "shortest_path"))

from graphs import bfs, random_kout_graph, sample_shortest_path  # noqa: E402
from make_data import make_puzzle, to_row  # noqa: E402
from reward import compute_score, parse_path  # noqa: E402
from score_passk import pass_at_k  # noqa: E402

# 0->1->3, 0->2->3 (two shortest paths of length 2), plus a longer 0->4->5->3.
EDGES = [(0, 1), (1, 3), (0, 2), (2, 3), (0, 4), (4, 5), (5, 3)]
GT = {"source": 0, "target": 3, "distance": 2,
      "edges_src": [a for a, _ in EDGES], "edges_dst": [b for _, b in EDGES]}


def ans(s):
    return f"<think>hmm</think> <answer>{s}</answer>"


def all_shortest_paths(adj, s, t):
    dist, _ = bfs(adj, s)
    d = dist[t]
    paths = []

    def rec(p):
        if len(p) == d + 1:
            if p[-1] == t:
                paths.append(tuple(p))
            return
        for v in adj.get(p[-1], ()):
            rec(p + [v])
    rec([s])
    return paths


LETTER_GT = {"source": "K", "target": "F", "distance": 2,
             "edges_src": ["K", "B", "K", "Q"], "edges_dst": ["B", "F", "Q", "F"]}


class TestReward(unittest.TestCase):
    def test_letter_labels(self):
        self.assertEqual(compute_score(None, ans("K -> B -> F"), LETTER_GT), 1.0)
        self.assertEqual(compute_score(None, ans("K -> Q -> F"), LETTER_GT), 1.0)
        self.assertEqual(compute_score(None, ans(" K -> B -> F "), LETTER_GT), 1.0)
        self.assertEqual(compute_score(None, ans("k -> b -> f"), LETTER_GT), 0.1)  # case-sensitive
        self.assertEqual(compute_score(None, ans("K -> F"), LETTER_GT), 0.1)

    def test_every_shortest_path_scores_one(self):
        self.assertEqual(compute_score(None, ans("0 -> 1 -> 3"), GT), 1.0)
        self.assertEqual(compute_score(None, ans("0 -> 2 -> 3"), GT), 1.0)

    def test_wrong_but_well_formed(self):
        self.assertEqual(compute_score(None, ans("0 -> 4 -> 5 -> 3"), GT), 0.1)  # valid, too long
        self.assertEqual(compute_score(None, ans("0 -> 5 -> 3"), GT), 0.1)       # 0->5 not an edge
        self.assertEqual(compute_score(None, ans("1 -> 3"), GT), 0.1)            # wrong start
        self.assertEqual(compute_score(None, ans("0 -> 1"), GT), 0.1)            # wrong end
        self.assertEqual(compute_score(None, ans("3 -> 1 -> 0"), GT), 0.1)       # edges reversed

    def test_unparseable(self):
        self.assertEqual(compute_score(None, "no tags 0 -> 1 -> 3", GT), 0.0)
        self.assertEqual(compute_score(None, ans("0, 1, 3"), GT), 0.0)
        self.assertEqual(compute_score(None, ans(""), GT), 0.0)
        self.assertEqual(compute_score(None, ans("0 -> -> 3"), GT), 0.0)

    def test_last_answer_is_graded(self):
        s = ans("0 -> 4 -> 5 -> 3") + ans("0 -> 2 -> 3")
        self.assertEqual(compute_score(None, s, GT), 1.0)

    def test_whitespace_tolerant(self):
        self.assertEqual(parse_path(" 0->1 ->  3 "), ["0", "1", "3"])
        self.assertEqual(parse_path(" K ->B-> F "), ["K", "B", "F"])
        self.assertEqual(compute_score(None, "<answer>\n0 -> 1 -> 3\n</answer>", GT), 1.0)

    def test_trailing_penalty(self):
        s = ans("0 -> 1 -> 3") + "x" * 1000
        self.assertAlmostEqual(compute_score(None, s, GT, trailing_penalty=0.5), 0.5)
        self.assertEqual(compute_score(None, s, GT), 1.0)

    def test_accepts_numpy_ground_truth(self):
        import numpy as np
        gt = {**GT, "edges_src": np.array(GT["edges_src"]), "edges_dst": np.array(GT["edges_dst"])}
        self.assertEqual(compute_score(None, ans("0 -> 2 -> 3"), gt), 1.0)


class TestGraphs(unittest.TestCase):
    def test_counts_and_uniform_sampling(self):
        adj = {0: [1, 2, 4], 1: [3], 2: [3], 4: [5], 5: [3]}
        dist, cnt = bfs(adj, 0)
        self.assertEqual((dist[3], cnt[3]), (2, 2))
        rng = random.Random(0)
        draws = [tuple(sample_shortest_path(adj, 0, 3, rng)) for _ in range(4000)]
        frac = draws.count((0, 1, 3)) / len(draws)
        self.assertTrue(0.45 < frac < 0.55, frac)

    def test_bfs_count_matches_enumeration(self):
        rng = random.Random(1)
        for _ in range(50):
            adj = random_kout_graph(10, rng)
            for s, t in itertools.permutations(range(10), 2):
                dist, cnt = bfs(adj, s)
                if t in dist:
                    self.assertEqual(cnt[t], len(all_shortest_paths(adj, s, t)))

    def test_kout_degrees(self):
        adj = random_kout_graph(14, random.Random(2))
        for u, vs in adj.items():
            self.assertIn(len(vs), (1, 2, 3))
            self.assertNotIn(u, vs)
            self.assertEqual(len(set(vs)), len(vs))


class TestPuzzles(unittest.TestCase):
    def test_puzzle_is_consistent_and_ref_path_scores_one(self):
        rng = random.Random(3)
        for d in (2, 3, 4, 5, 6):
            for _ in range(20):
                p = make_puzzle(d, rng, 12, 16)
                row = to_row(p, "test", 0)
                gt = row["reward_model"]["ground_truth"]
                ref = " -> ".join(map(str, row["extra_info"]["ref_path"]))
                self.assertEqual(compute_score(None, ans(ref), gt), 1.0)
                prompt = row["prompt"][0]["content"]
                self.assertIn(f"from node {p['source']} to node {p['target']}", prompt)
                edge_line = prompt.split("\n")[1]
                self.assertEqual(edge_line.count(" -> "), len(p["edges"]))
                self.assertTrue(prompt.endswith("Solution:\n"))
                nodes = {a for a, _ in p["edges"]} | {b for _, b in p["edges"]} | {p["source"], p["target"]}
                self.assertTrue(all(len(x) == 1 and x.isupper() for x in nodes))
                # The prompt's example path uses letters absent from the graph, so
                # copying it is never a correct path.
                ex = p["example_nodes"]
                self.assertFalse(set(ex) & nodes)
                self.assertIn(f"<answer> {' -> '.join(ex)} </answer>", prompt)
                self.assertEqual(compute_score(None, ans(" -> ".join(ex)), gt), 0.1)


class TestPassAtK(unittest.TestCase):
    def test_values(self):
        self.assertEqual(pass_at_k(10, 0, 5), 0.0)
        self.assertEqual(pass_at_k(10, 10, 1), 1.0)
        self.assertAlmostEqual(pass_at_k(10, 3, 1), 0.3)
        self.assertEqual(pass_at_k(10, 6, 5), 1.0)  # n - c < k


if __name__ == "__main__":
    unittest.main()
