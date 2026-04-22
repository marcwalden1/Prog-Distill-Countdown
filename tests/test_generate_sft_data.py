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
    """Default: all responses kept. With --filter_correct_only: only score==1.0."""

    def _run_main(self, tmp_dir, rows, llm_response_texts, filter_correct_only=False):
        """
        Run generate_sft_data.main() with mocked vLLM and return the output DataFrame.

        llm_response_texts: list of lists, one per row, each sub-list contains
                            the raw generation texts for that prompt.
        filter_correct_only: pass --filter_correct_only to the script under test.
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
            argv = [
                "generate_sft_data.py",
                "--teacher_checkpoint_path", "/fake/ckpt",
                "--output_path", output_path,
                "--data_path", data_path,
                "--n_responses", "4",
                "--max_length", "512",
            ]
            if filter_correct_only:
                argv.append("--filter_correct_only")
            sys.argv = argv
            generate_sft_data.main()

        return pd.read_parquet(output_path)

    # -- Default behaviour: keep all responses -------------------------------

    def test_default_keeps_all_responses(self):
        """Default: correct + format-only-wrong + no-answer all kept."""
        with tempfile.TemporaryDirectory() as d:
            correct_text = "3+4+5 = 12</think>\n<answer> 3+4+5 </answer>"
            wrong_text   = "3+4+5 = 100</think>\n<answer> 3+4+100 </answer>"
            no_ans_text  = "no idea</think>"

            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[correct_text, wrong_text, no_ans_text]]

            out_df = self._run_main(d, rows, llm_texts)
            self.assertEqual(len(out_df), 3, "Expected all 3 responses kept by default")

    def test_default_all_wrong_still_kept(self):
        """With no correct responses, default still keeps all wrong ones."""
        with tempfile.TemporaryDirectory() as d:
            wrong_text = "I give up</think>\n<answer> 99 </answer>"
            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[wrong_text, wrong_text]]
            out_df = self._run_main(d, rows, llm_texts)
            self.assertEqual(len(out_df), 2)

    # -- --filter_correct_only behaviour -------------------------------------

    def test_filter_correct_solutions_only_kept(self):
        """With --filter_correct_only: only score==1.0 responses kept."""
        with tempfile.TemporaryDirectory() as d:
            correct_text = "3+4+5 = 12</think>\n<answer> 3+4+5 </answer>"
            wrong_text   = "3+4+5 = 100</think>\n<answer> 3+4+100 </answer>"

            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[correct_text, wrong_text]]

            out_df = self._run_main(d, rows, llm_texts, filter_correct_only=True)

            self.assertEqual(len(out_df), 1, "Expected exactly 1 correct response")
            self.assertIn("prompt", out_df.columns)
            self.assertIn("response", out_df.columns)
            self.assertIn("<answer> 3+4+5 </answer>", out_df["response"].iloc[0])

    def test_filter_no_correct_solutions_gives_empty_df(self):
        """--filter_correct_only with no correct responses → empty parquet."""
        with tempfile.TemporaryDirectory() as d:
            wrong_text = "I give up</think>\n<answer> 99 </answer>"
            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[wrong_text, wrong_text]]
            out_df = self._run_main(d, rows, llm_texts, filter_correct_only=True)
            self.assertEqual(len(out_df), 0)

    # -- Behaviour shared across both modes ----------------------------------

    def test_multiple_correct_responses_all_kept(self):
        """Both correct responses from the same prompt appear as separate rows."""
        with tempfile.TemporaryDirectory() as d:
            c1 = "3+4+5=12</think>\n<answer> 3+4+5 </answer>"
            c2 = "3+(4+5)=12</think>\n<answer> 3+(4+5) </answer>"
            rows = [("prompt1", 12, [3, 4, 5])]
            llm_texts = [[c1, c2]]
            # Use filter_correct_only to isolate the "multiple correct" behaviour
            out_df = self._run_main(d, rows, llm_texts, filter_correct_only=True)
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

    def test_filter_multi_prompt_coverage(self):
        """3 prompts × 1 response each, 2 correct: with --filter_correct_only → 2 rows."""
        with tempfile.TemporaryDirectory() as d:
            c = "3+4+5=12</think>\n<answer> 3+4+5 </answer>"
            w = "nope</think>\n<answer> 99 </answer>"
            rows = [
                ("p1", 12, [3, 4, 5]),
                ("p2", 12, [3, 4, 5]),
                ("p3", 12, [3, 4, 5]),
            ]
            llm_texts = [[c], [w], [c]]
            out_df = self._run_main(d, rows, llm_texts, filter_correct_only=True)
            self.assertEqual(len(out_df), 2)

    def test_default_multi_prompt_keeps_all(self):
        """Same 3×1 setup, default mode keeps all 3 responses regardless of score."""
        with tempfile.TemporaryDirectory() as d:
            c = "3+4+5=12</think>\n<answer> 3+4+5 </answer>"
            w = "nope</think>\n<answer> 99 </answer>"
            rows = [
                ("p1", 12, [3, 4, 5]),
                ("p2", 12, [3, 4, 5]),
                ("p3", 12, [3, 4, 5]),
            ]
            llm_texts = [[c], [w], [c]]
            out_df = self._run_main(d, rows, llm_texts)
            self.assertEqual(len(out_df), 3)


class TestTruncation(unittest.TestCase):
    """
    truncate_to_final_attempt keeps text from just after the second-to-last
    `</answer>` through the end of the response.
    """

    def test_three_answers_keeps_after_penultimate(self):
        r = "<think>r</think>A<answer>1</answer>B<answer>2</answer>C<answer>3</answer>"
        self.assertEqual(
            generate_sft_data.truncate_to_final_attempt(r),
            "C<answer>3</answer>",
        )

    def test_two_answers_keeps_after_first(self):
        r = "X<answer>a</answer>Y<answer>b</answer>"
        self.assertEqual(
            generate_sft_data.truncate_to_final_attempt(r),
            "Y<answer>b</answer>",
        )

    def test_single_answer_unchanged(self):
        r = "only <answer>once</answer>"
        self.assertEqual(generate_sft_data.truncate_to_final_attempt(r), r)

    def test_no_answers_unchanged(self):
        r = "no answer tags here"
        self.assertEqual(generate_sft_data.truncate_to_final_attempt(r), r)

    def test_trailing_text_after_final_answer_preserved(self):
        r = "<answer>1</answer><answer>2</answer>trailing"
        self.assertEqual(
            generate_sft_data.truncate_to_final_attempt(r),
            "<answer>2</answer>trailing",
        )

    def test_empty_answer_tags_handled(self):
        r = "<answer></answer><answer></answer>"
        self.assertEqual(
            generate_sft_data.truncate_to_final_attempt(r),
            "<answer></answer>",
        )


class TestTruncateFlagIntegration(unittest.TestCase):
    """
    End-to-end: with --truncate on the CLI, the saved parquet's response column
    contains only the final-attempt substring, not the full teacher output.
    """

    def test_truncate_flag_truncates_saved_response(self):
        with tempfile.TemporaryDirectory() as d:
            data_path = os.path.join(d, "train.parquet")
            output_path = os.path.join(d, "sft_out.parquet")
            make_sample_parquet(data_path, [("prompt1", 12, [3, 4, 5])])

            full = (
                "<think>let me think</think>\n"
                "<answer> 3+4+6 </answer> no, 3+4+6=13\n"
                "<answer> 3+5+5 </answer> no, 3+5+5=13\n"
                "<answer> 3+4+5 </answer>"
            )
            expected_truncated = full[full.rindex("</answer>", 0, full.rindex("</answer>")) + len("</answer>"):]
            # Sanity: truncation drops the FIRST attempt (3+4+6) entirely.
            # It preserves the trailing verification of the 2nd-to-last attempt
            # ("no, 3+5+5=13") because that's the reasoning leading into the final answer.
            self.assertNotIn("3+4+6", expected_truncated)
            self.assertIn("<answer> 3+4+5 </answer>", expected_truncated)

            fake_tok = _fake_tokenizer()
            fake_outputs = _fake_llm_output([[full]])
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
                    "--n_responses", "1",
                    "--max_length", "512",
                    "--truncate",
                ]
                generate_sft_data.main()

            out_df = pd.read_parquet(output_path)
            self.assertEqual(len(out_df), 1)
            saved = out_df["response"].iloc[0]
            # Saved response should end with the final answer block and NOT
            # contain the FIRST (earliest) wrong attempt — that's dropped.
            self.assertNotIn("3+4+6", saved)
            self.assertIn("<answer> 3+4+5 </answer>", saved)
            # The saved response equals the truncation of the full response
            self.assertEqual(saved, expected_truncated)


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
