import math
import re
from types import SimpleNamespace
import unittest

from frozen_inference.executor import Rejected
from frozen_inference.neural import (FrozenSequenceScorer, activation_transport,
                                     choice_views, contrast_vector, counterfactual_score)
try:
    import torch
except ImportError:
    torch = None


class CounterfactualTests(unittest.TestCase):
    question = "Which color?\n(A) red\n(B) blue\n(C) green"

    def test_mapping_back_to_original_answer(self):
        def scorer(text, labels):
            blue = re.search(r"\(([A-Z])\) blue", text)[1]
            return {label: -0.1 if label == blue else -3.0 for label in labels}
        result = counterfactual_score(self.question, scorer)
        self.assertEqual(result["prediction"], "(B)")
        self.assertTrue(result["unanimous"])
        self.assertEqual(len(result["views"]), 3)

    def test_label_bias_breaks_mapped_agreement(self):
        result = counterfactual_score(self.question, lambda text, labels: {k: -0.1 if k == "A" else -3 for k in labels})
        self.assertFalse(result["unanimous"])
        self.assertAlmostEqual(result["log_margin"], 0)

    def test_dependent_options_and_missing_scores_rejected(self):
        for question in ["q\n(A) red\n(B) both red and blue", "Use option A.\n(A) red\n(B) blue",
                         "q\n(A) red\n(B) all of the above", "q\n(A) red\n(B) blue\nExplain"]:
            with self.subTest(question=question), self.assertRaises(Rejected):
                choice_views(question)
        with self.assertRaises(Rejected):
            counterfactual_score(self.question, lambda text, labels: {"A": -1})
        with self.assertRaises(Rejected):
            counterfactual_score(self.question, lambda text, labels: {k: float("nan") for k in labels})


@unittest.skipIf(torch is None, "optional PyTorch not installed")
class NeuralTests(unittest.TestCase):
    def make_model(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3, 3, bias=False)
            def forward(self, value):
                return self.layer(value)
        return Model().eval()

    def test_transport_changes_activations_not_weights_and_cleans_up(self):
        model = self.make_model()
        x = torch.ones(1, 4, 3)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        original = model(x).detach().clone()
        with activation_transport(model, {"layer": torch.ones(3)}, scale=0.25):
            output = model(x)
        self.assertTrue(torch.allclose(output[:, -1], original[:, -1] + 0.25))
        self.assertTrue(torch.equal(output[:, :-1], original[:, :-1]))
        self.assertTrue(torch.equal(model(x), original))
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_disabled_gate_and_error_cleanup(self):
        model = self.make_model()
        x = torch.ones(1, 2, 3)
        original = model(x).detach().clone()
        with activation_transport(model, {"layer": torch.ones(3)}, gate=lambda: False):
            self.assertTrue(torch.equal(model(x), original))
        with self.assertRaises(ValueError):
            with activation_transport(model, {"layer": torch.ones(4)}):
                model(x)
        self.assertEqual(len(model.layer._forward_hooks), 0)
        with self.assertRaises(AttributeError):
            with activation_transport(model, {"layer": torch.ones(3), "missing": torch.ones(3)}):
                pass
        self.assertEqual(len(model.layer._forward_hooks), 0)

    def test_padded_or_multi_example_batch_is_rejected(self):
        model = self.make_model()
        with self.assertRaises(ValueError):
            with activation_transport(model, {"layer": torch.ones(3)}):
                model(torch.ones(2, 2, 3))

    def test_contrast_is_mean_not_training(self):
        positive = torch.tensor([[3.0, 5.0], [5.0, 9.0]], requires_grad=True)
        negative = torch.tensor([[1.0, 1.0], [1.0, 3.0]])
        result = contrast_vector(positive, negative)
        self.assertTrue(torch.equal(result, torch.tensor([3.0, 5.0])))
        self.assertFalse(result.requires_grad)
        self.assertIsNone(positive.grad)
        with self.assertRaises(ValueError):
            contrast_vector(positive[:1], negative[:1])

    def test_multitoken_scoring_matches_manual_likelihood(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(8, 8)
                with torch.no_grad():
                    self.embed.weight.copy_(torch.arange(64).reshape(8, 8).float() / 20)
            def get_input_embeddings(self):
                return self.embed
            def forward(self, input_ids, attention_mask, use_cache):
                return SimpleNamespace(logits=self.embed(input_ids))
        class Tokenizer:
            pad_token_id, eos_token_id = 0, 7
            def apply_chat_template(self, messages, **kwargs):
                assert messages[0]["role"] == "user" and len(messages) == 1
                return [1, 2]
            def encode(self, label, **kwargs):
                return {"A": [3], "B": [4, 5]}[label]
            def decode(self, ids, **kwargs):
                return {(3,): "A", (4, 5): "B"}[tuple(ids)]
        model = Model()
        weights = model.embed.weight.detach().clone()
        scorer = FrozenSequenceScorer(model, Tokenizer())
        scores = scorer("Question", ["A", "B"])
        expected_a = float(weights[2].log_softmax(-1)[3])
        expected_b = float(weights[2].log_softmax(-1)[4] + weights[4].log_softmax(-1)[5])
        self.assertAlmostEqual(scores["A"], expected_a, places=6)
        self.assertAlmostEqual(scores["B"], expected_b, places=6)
        self.assertTrue(torch.equal(model.embed.weight, weights))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertEqual(scorer.telemetry[0]["prefill_tokens"], 4)
        self.assertEqual(scorer.telemetry[0]["continuation_tokens"], 3)
        with self.assertRaises(Rejected):
            FrozenSequenceScorer(model, Tokenizer(), max_context=2)("q", ["A", "B"])


if __name__ == "__main__":
    unittest.main()
