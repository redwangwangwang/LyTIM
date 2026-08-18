import unittest

import torch

from models.utils.lytim_controller import (
    LyTIMController,
    TRANSITION_NEW,
    TRANSITION_PERSISTENT,
    TRANSITION_RESOLVED,
    TRANSITION_STABLE_NEGATIVE,
    labels_to_probabilities,
    monotonic_acceptance_mask,
    transition_targets_from_labels,
)


class LyTIMControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.controller = LyTIMController(
            hidden_size=32,
            controller_dim=16,
            num_prompt_tokens=4,
        )

    def test_transition_targets(self):
        previous = torch.tensor([[2, 1, 1, 2, 0, 1]])
        current = torch.tensor([[1, 2, 1, 2, 1, 3]])
        targets, valid = transition_targets_from_labels(previous, current)
        self.assertEqual(targets[0, 0].item(), TRANSITION_NEW)
        self.assertEqual(targets[0, 1].item(), TRANSITION_RESOLVED)
        self.assertEqual(targets[0, 2].item(), TRANSITION_PERSISTENT)
        self.assertEqual(targets[0, 3].item(), TRANSITION_STABLE_NEGATIVE)
        self.assertFalse(valid[0, 4].item())
        self.assertFalse(valid[0, 5].item())

    def test_aligned_energy_is_lower(self):
        batch, diseases = 2, 14
        previous_labels = torch.full((batch, diseases), 2)
        current_labels = torch.full((batch, diseases), 2)
        previous_prob = labels_to_probabilities(previous_labels)
        current_prob = labels_to_probabilities(current_labels)

        transitions = torch.zeros(batch, diseases, 4)
        transitions[..., TRANSITION_STABLE_NEGATIVE] = 1.0
        aligned = self.controller.compute_energy(
            report_probabilities=current_prob,
            image_probabilities=current_prob,
            progression_probabilities=transitions,
            previous_probabilities=previous_prob,
            backward_probabilities=previous_prob,
        )

        contradicted = current_prob.roll(shifts=1, dims=-1)
        misaligned = self.controller.compute_energy(
            report_probabilities=contradicted,
            image_probabilities=current_prob,
            progression_probabilities=transitions,
            previous_probabilities=previous_prob,
            backward_probabilities=contradicted,
        )
        self.assertTrue(torch.all(aligned.total < misaligned.total))
        self.assertTrue(torch.all(aligned.total >= 0))

    def test_monotonic_acceptance_and_rollback_rule(self):
        current = torch.tensor([0.50, 0.50, 0.20])
        candidate = torch.tensor([0.40, 0.495, 0.10])
        active = torch.tensor([True, True, False])
        accept = monotonic_acceptance_mask(current, candidate, epsilon=0.01, active=active)
        self.assertEqual(accept.tolist(), [True, False, False])

    def test_controller_shapes_and_gradients(self):
        previous = torch.randn(3, 5, 32)
        current = torch.randn(3, 5, 32)
        image_outputs = self.controller.encode_images(previous, current)
        self.assertEqual(image_outputs["image_state_logits"].shape, (3, 14, 4))
        self.assertEqual(image_outputs["progression_logits"].shape, (3, 14, 4))

        report_hidden = torch.randn(3, 7, 32)
        report_vector, report_logits = self.controller.encode_report(report_hidden)
        backward_logits = self.controller.predict_backward_state(
            report_vector, image_outputs["progression_vector"]
        )
        self.assertEqual(report_logits.shape, (3, 14, 4))
        self.assertEqual(backward_logits.shape, (3, 14, 4))

        difference_prompt = torch.randn(3, 4, 32)
        fact_errors = torch.rand(3, 14)
        gated, gate = self.controller.gate_refinement_prompt(
            difference_prompt, fact_errors, report_vector
        )
        self.assertEqual(gated.shape, difference_prompt.shape)
        self.assertEqual(gate.shape, (3, 14))
        self.assertTrue(torch.all((gate >= 0) & (gate <= 1)))

        loss = gated.square().mean() + report_logits.square().mean() + backward_logits.square().mean()
        loss.backward()
        self.assertIsNotNone(self.controller.fact_embeddings.grad)


if __name__ == "__main__":
    unittest.main()
