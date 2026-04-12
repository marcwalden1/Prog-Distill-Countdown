"""
Unit tests for scripts/generate_sft_data.py

Run with:
    source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
    conda activate verl
    python3 -m unittest tests/test_generate_sft_data.py -v
"""
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Add repo root to path so we can import grader_utils and the script under test
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# We import generate_sft_data as a module by loading it via importlib after
# adding scripts/ to the path — this lets us call main() in tests.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "generate_sft_data",
    os.path.join(REPO_ROOT, "scripts", "generate_sft_data.py"),
)
generate_sft_data = importlib.util.module_from_spec(_spec)
sys.modules["generate_sft_data"] = generate_sft_data  # needed for patch() to resolve
_spec.loader.exec_module(generate_sft_data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample_parquet(path, rows):
    """
    Write a minimal training parquet to *path*.

    Each row is a tuple: (prompt_content_str, target_int, numbers_list)
    """
    records = []
    for content, target, numbers in rows:
        records.append({
            "prompt": [{"role": "user", "content": content}],
            "reward_model": {"ground_truth": target},
            "extra_info": {"numbers": numbers},
        })
    pd.DataFrame(records).to_parquet(path, index=False)


def _fake_tokenizer(apply_result="<prompt>", eos_token="</s>"):
    tok = MagicMock()
    tok.eos_token = eos_token
    tok.apply_chat_template.return_value = apply_result
    return tok


def _fake_llm_output(texts):
    """Build a fake vLLM output list (one per prompt, each with len(texts) completions)."""
    outputs = []
    for prompt_texts in texts:
        completions = [MagicMock(text=t) for t in prompt_texts]
        out = MagicMock()
        out.outputs = completions
        outputs.append(out)
    return outputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPromptExtraction(unittest.TestCase):
    """generate_sft_data.py must extract plain strings from list-of-dict prompts."""

    def test_extracts_content_from_list_of_dicts(self):
        with tempfile.TemporaryDirectory() as d:
            parquet_path = os.path.join(d, "train.parquet")
            make_sample_parquet(parquet_path, [
                ("Using 3 4 5, make 12.", 12, [3, 4, 5]),
                ("Using 2 6, make 8.",     8,  [2, 6]),
            ])
            df = pd.read_parquet(parquet_path)
            prompts = []
            for row in df["prompt"]:
                content = row[-1]["content"] if isinstance(row[-1], dict) else row[-1]
                prompts.append(str(content))
            self.assertEqual(prompts[0], "Using 3 4 5, make 12.")
            self.assertEqual(prompts[1], "Using 2 6, make 8.")

    def test_prompt_is_plain_string_not_list(self):
        """SFTDataset wraps the string in a chat template — it must NOT be a list."""
        with tempfile.TemporaryDirectory() as d:
            parquet_path = os.path.join(d, "train.parquet")
            make_sample_parquet(parquet_path, [("some question", 5, [2, 3])])
            df = pd.read_parquet(parquet_path)
            for row in df["prompt"]:
                content = row[-1]["content"] if isinstance(row[-1], dict) else row[-1]
                self.assertIsInstance(content, str)
                self.assertNotIsInstance(content, list)


class TestFiltering(unittest.TestCase):
    """Only score==1.0 responses should be kept in the SFT parquet."""

    def _run_main(self, tmp_dir, rows, llm_response_texts):
        """
        Run generate_sft_data.main() with mocked vLLM and return the output DataFrame.

        llm_response_texts: list of lists, one per row, each sub-list contains
                            the raw generation texts for that prompt.
        """
        data_path = os.path.join(tmp_dir, "train.parquet")
        output_path = os.path.join(tmp_dir, "sft_out.parquet")
        make_sample_parquet(data_path, rows)

        fake_tok = _fake_tokenizer()
        fake_outputs = _fake_llm_output(llm_response_texts)
        fake_llm_instance = MagicMock()
        fake_llm_instance.generate.return_value = fake_outputs

        with patch("generate_sft_data.LLM", return_value=fake_llm_instance), \
             patch("generate_sft_data.SamplingParams", return_value=MagicMock()), \
             patch("generate_sft_data.AutoTokenizer") as mock_tok_cls:
            mock_tok_cls.from_pretrained.return_value = fake_tok
            sys.argv = [
                "generate_sft_data.py",
                "--teacher_checkpoint_path", "/fake/ckpt",
                "--output_path", output_path,
                "--data_path", data_path,
                "--n_responses", "4",
                "--max_length", "512",
            ]
            generate_sft_data.main()

        return pd.read_parquet(output_path)

    def test_correct_solutions_only_kept(self):
        """
        Prompt: target=12, numbers=[3,4,5].
        Give one correct response (3+4+5=12 inside <answer> tags) and one wrong one.
        Only the correct one should appear in the output.
        """
        with tempfile.TemporaryDirectory() as d:
            # The full model output is "Let me solve...<think>" prefix + completion text
            # generate_sft_data.py prepends "Let me solve this step by step.\n<think>"
            # then scores the full concatenation. We need valid <answer> tags.
            correct_text = "3+4+5 = 12</think>\n<answer> 3+4+5 </answer>"
            wrong_text   = "3+4+5 = 100</think>\n<answer> 3+4+100 </answer>"

            rows = [("prompt1", 12, [3, 4, 5])]
            # Two completions for the one prompt
            llm_texts = [[correct_text, wrong_text]]

            out_df = self._run_main(d, rows, llm_texts)

            self.assertEqual(len(out_df), 1, "Expected exactly 1 correct response")
            self.assertIn("prompt", out_df.columns)
            self.assertIn("response", out_df.columns)
            # Response must be the full generation including the prefix
            self.assertIn("<answer> 3+4+5 </answer>", out_df["response"].iloc[0])

    def test_no_correct_solutions_gives_empty_df(self):
        """If no response is correct the output parquet should have 0 rows."""
        with tempfile.TemporaryDirectory() as d:
            wrong_text = "I give up</think>\n<answer> 99 </answer>"
            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[wrong_text, wrong_text]]
            out_df = self._run_main(d, rows, llm_texts)
            self.assertEqual(len(out_df), 0)

    def test_multiple_correct_responses_all_kept(self):
        """Both correct responses from the same prompt should appear as separate rows."""
        with tempfile.TemporaryDirectory() as d:
            c1 = "3+4+5=12</think>\n<answer> 3+4+5 </answer>"
            c2 = "3+(4+5)=12</think>\n<answer> 3+(4+5) </answer>"
            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[c1, c2]]
            out_df = self._run_main(d, rows, llm_texts)
            self.assertEqual(len(out_df), 2)
            self.assertEqual(out_df["prompt"].iloc[0], out_df["prompt"].iloc[1],
                             "Both rows should have the same prompt")

    def test_output_schema(self):
        """Output parquet must have exactly 'prompt' and 'response' string columns."""
        with tempfile.TemporaryDirectory() as d:
            correct_text = "3+4+5=12</think>\n<answer> 3+4+5 </answer>"
            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[correct_text]]
            out_df = self._run_main(d, rows, llm_texts)
            self.assertSetEqual(set(out_df.columns), {"prompt", "response"})
            self.assertTrue(all(isinstance(v, str) for v in out_df["prompt"]))
            self.assertTrue(all(isinstance(v, str) for v in out_df["response"]))

    def test_multi_prompt_coverage(self):
        """
        3 prompts, only 2 have correct solutions.
        Coverage should be 2/3 correct prompts.
        """
        with tempfile.TemporaryDirectory() as d:
            c = "3+4+5=12</think>\n<answer> 3+4+5 </answer>"
            w = "nope</think>\n<answer> 99 </answer>"
            rows = [
                ("p1", 12, [3, 4, 5]),
                ("p2", 12, [3, 4, 5]),
                ("p3", 12, [3, 4, 5]),
            ]
            # prompt 1: 1 correct, prompt 2: 0 correct, prompt 3: 1 correct
            llm_texts = [[c], [w], [c]]
            out_df = self._run_main(d, rows, llm_texts)
            self.assertEqual(len(out_df), 2)


class TestGraderUtilsIntegration(unittest.TestCase):
    """
    Verify that grader_utils.compute_score behaves correctly for the cases
    generate_sft_data.py depends on.
    """

    def setUp(self):
        from grader_utils import compute_score
        self.compute_score = compute_score

    def _score(self, solution_str, target, numbers):
        return self.compute_score(
            data_source=None,
            solution_str=solution_str,
            ground_truth=target,
            extra_info={"numbers": numbers},
            verbose=False,
        )

    def test_correct_answer_scores_1(self):
        s = "Let me solve this step by step.\n<think>stuff</think>\n<answer> 3+4+5 </answer>"
        self.assertEqual(self._score(s, 12, [3, 4, 5]), 1.0)

    def test_wrong_answer_scores_format(self):
        s = "Let me solve this step by step.\n<think>stuff</think>\n<answer> 3+4+5 </answer>"
        # correct format but wrong target
        self.assertEqual(self._score(s, 99, [3, 4, 5]), 0.1)

    def test_no_answer_tag_scores_0(self):
        s = "Let me solve this step by step.\n<think>I don't know</think>"
        self.assertEqual(self._score(s, 12, [3, 4, 5]), 0)

    def test_last_answer_tag_used(self):
        """Multiple <answer> tags — only the last one should count."""
        s = "<answer> 99 </answer> more thinking <answer> 3+4+5 </answer>"
        self.assertEqual(self._score(s, 12, [3, 4, 5]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
