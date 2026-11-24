"""Reward for the shortest-path task (verl custom_reward_function).

Use with train_grpo.sh:
    REWARD_FN_PATH=$PWD/tasks/shortest_path/reward.py DATA_SOURCE=spath-simple ...

ground_truth (parquet reward_model.ground_truth) is a dict:
    {"source": str, "target": str, "distance": int,
     "edges_src": [str], "edges_dst": [str]}     (node labels, e.g. "K")

Score (same scale as grader_utils.compute_score for Countdown):
    1.0  the LAST <answer> is a node sequence that starts at source, ends at
         target, follows only existing directed edges, and has exactly
         `distance` edges. ANY shortest path qualifies -- no reference path is
         compared. (A length-`distance` walk is automatically simple: a repeated
         node would leave a shorter s->t walk, contradicting `distance`.)
    0.1  the answer parses as a node sequence "K -> B -> ..." but is wrong.
    0.0  no <answer> tag, or its content does not parse.

Self-contained (stdlib only): verl imports this file by path.
"""

import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NODE_SEQ_RE = re.compile(r"^\s*[A-Za-z0-9]+(\s*->\s*[A-Za-z0-9]+)*\s*$")


def extract_answer(solution_str):
    matches = _ANSWER_RE.findall(solution_str)
    return matches[-1].strip() if matches else None


def parse_path(answer):
    """'K -> B -> F' -> ['K', 'B', 'F']; None if not a plain node sequence.

    Labels are compared as strings (letters in the current data; ints also work)."""
    if answer is None or not _NODE_SEQ_RE.match(answer):
        return None
    return [x.strip() for x in answer.split("->")]


def is_shortest_path(path, gt):
    src = [str(x) for x in gt["edges_src"]]
    dst = [str(x) for x in gt["edges_dst"]]
    edges = set(zip(src, dst))
    return (len(path) == int(gt["distance"]) + 1
            and path[0] == str(gt["source"])
            and path[-1] == str(gt["target"])
            and all((a, b) in edges for a, b in zip(path, path[1:])))


def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  format_score=0.1, score=1.0, trailing_penalty=0.0, **kwargs):
    path = parse_path(extract_answer(solution_str))
    if path is None:
        return 0.0
    base = score if is_shortest_path(path, ground_truth) else format_score
    if trailing_penalty > 0:
        # Same rule as grader_utils: deduct per 1000 chars after the final </answer>.
        tail = solution_str[solution_str.rfind("</answer>") + len("</answer>"):].strip()
        base = max(base - trailing_penalty * len(tail) / 1000.0, 0.0)
    return base
