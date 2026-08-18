"""LyTIM: Lyapunov-guided monotonic mutual refinement for TIM.

This implementation keeps the original TIM Stage-II model intact and subclasses it
with four additions:

1. a multi-view clinical surrogate energy;
2. a fact-level trust gate over refinement prompts;
3. monotonic accept/rollback inference;
4. an adaptive stopping head.

No extra dataset fields or annotations are required.  When Stage-I reports are not
present in a batch, the frozen Stage-I path generates them online.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F

from models.model_stage2 import LongitudinalR2GenGPT as TIMStage2
from models.utils.lytim_controller import (
    EnergyBreakdown,
    LyTIMController,
    masked_mean,
    monotonic_acceptance_mask,
    safe_cross_entropy,
    transition_targets_from_labels,
)


class LongitudinalR2GenGPT(TIMStage2):
    """TIM Stage II augmented with Lyapunov-guided clinical refinement."""

    def __init__(self, args: Any) -> None:
        # The upstream loader expects a non-existent ``model`` key.  Defer loading and
        # accept both TIM's ``state_dict`` checkpoints and older ``model`` checkpoints.
        delta_file = getattr(args, "delta_file", None)
        setattr(args, "delta_file", None)
        try:
            super().__init__(args)
        finally:
            setattr(args, "delta_file", delta_file)

        self.args = args
        self.hparams.delta_file = delta_file
        # Upstream prompt_wrap refers to ``prior_prompt`` while defining
        # ``prompt_prior``.  Keep an alias so both TIM and LyTIM checkpoints work.
        self.prior_prompt = self.prompt_prior

        prompt_tokens = int(self.triplet_encoder.base_prompt_embedding.shape[0])
        requested_prompt_tokens = int(self._arg("lytim_num_prompt_tokens", prompt_tokens))
        if requested_prompt_tokens != prompt_tokens:
            raise ValueError(
                "--lytim_num_prompt_tokens must match TIM's TripletEncoder prompt count "
                f"({prompt_tokens})."
            )

        self.lytim_controller = LyTIMController(
            hidden_size=self.llama_model.config.hidden_size,
            controller_dim=int(self._arg("lytim_controller_dim", 512)),
            num_prompt_tokens=prompt_tokens,
            state_weight=float(self._arg("lytim_state_weight", 1.0)),
            change_weight=float(self._arg("lytim_change_weight", 1.0)),
            backward_weight=float(self._arg("lytim_backward_weight", 1.0)),
        )

        if delta_file:
            self._load_tim_delta(delta_file)

        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        print(f"LyTIM controller initialized: {trainable:,}/{total:,} trainable parameters")

    def _arg(self, name: str, default: Any) -> Any:
        return getattr(self.args, name, getattr(self.hparams, name, default))

    def _load_tim_delta(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported checkpoint format at {path!r}.")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if not isinstance(state_dict, dict):
            raise TypeError(f"Checkpoint {path!r} does not contain a state dictionary.")
        incompatible = self.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded TIM delta from {path}; "
            f"missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}"
        )

    def train(self, mode: bool = True):
        module = super().train(mode)
        # CheXbert provides fixed pseudo-labels and must never activate dropout.
        self.chexbert_metrics.eval()
        return module

    # ------------------------------------------------------------------
    # Frozen TIM representation and generation helpers
    # ------------------------------------------------------------------
    def _encode_longitudinal_images(self, samples: dict[str, Any]) -> dict[str, torch.Tensor]:
        previous_images = samples["prev_image"]
        current_images = samples["curr_image"]
        with torch.no_grad():
            previous_embed = self.layer_norm(self.encode_img(previous_images))
            current_embed = self.layer_norm(self.encode_img(current_images))
            batch_size = previous_embed.shape[0]
            perception_frames = self.perception_frame.expand(batch_size, -1, -1, -1, -1)
            video = torch.stack(
                [
                    previous_images[0],
                    perception_frames[:, 0],
                    perception_frames[:, 1],
                    current_images[0],
                ],
                dim=1,
            )
            perception = self.video_encoder(video)[:, 1:3, :, :]
            alpha = torch.sigmoid(self.perception_agg(perception).squeeze(1))
            perception = alpha * perception[:, 0, :, :] + (1.0 - alpha) * perception[:, 1, :, :]
            perception = self.video_layer_norm(self.video_linear(perception))
        return {
            "previous_embed": previous_embed.detach(),
            "current_embed": current_embed.detach(),
            "perception_embed": perception.detach(),
        }

    def _bos(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = torch.full(
            (batch_size, 1),
            fill_value=self.llama_tokenizer.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        return self.embed_tokens(token_ids), torch.ones_like(token_ids)

    def _generation_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "pad_token_id": self.llama_tokenizer.pad_token_id,
            "num_beams": int(self.hparams.beam_size),
            "do_sample": bool(self.hparams.do_sample),
            "min_new_tokens": int(self.hparams.min_new_tokens),
            "max_new_tokens": int(self.hparams.max_new_tokens),
            "repetition_penalty": float(self.hparams.repetition_penalty),
            "length_penalty": float(self.hparams.length_penalty),
            "no_repeat_ngram_size": int(getattr(self.hparams, "no_repeat_ngram_size", 0)),
            "use_cache": True,
        }
        if kwargs["do_sample"]:
            temperature = float(getattr(self.hparams, "temperature", 1.0))
            kwargs["temperature"] = max(temperature, 1e-5)
            kwargs["top_p"] = float(getattr(self.hparams, "top_p", 1.0))
        return kwargs

    def _decode_batch(self, token_ids: torch.Tensor) -> list[str]:
        return [self.decode(sequence) for sequence in token_ids]

    def _generate_stage1_current(
        self,
        image_bundle: dict[str, torch.Tensor],
        previous_reports: list[str],
    ) -> list[str]:
        current_embed = image_bundle["current_embed"]
        prompt_embed, prompt_mask = self.prompt_wrap(
            current_embed,
            image_bundle["perception_embed"],
            previous_reports,
            timepoint="curr",
        )
        bos_embed, bos_mask = self._bos(current_embed.shape[0], current_embed.device)
        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=torch.cat([bos_embed, prompt_embed], dim=1),
                attention_mask=torch.cat([bos_mask, prompt_mask], dim=1),
                **self._generation_kwargs(),
            )
        return self._decode_batch(outputs)

    def _generate_backward_prior(
        self,
        image_bundle: dict[str, torch.Tensor],
        current_reports: list[str],
    ) -> list[str]:
        previous_embed = image_bundle["previous_embed"]
        prompt_embed, prompt_mask = self.prompt_wrap(
            previous_embed,
            image_bundle["perception_embed"],
            current_reports,
            timepoint="prior",
        )
        bos_embed, bos_mask = self._bos(previous_embed.shape[0], previous_embed.device)
        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=torch.cat([bos_embed, prompt_embed], dim=1),
                attention_mask=torch.cat([bos_mask, prompt_mask], dim=1),
                **self._generation_kwargs(),
            )
        return self._decode_batch(outputs)

    def _generate_refined_current(
        self,
        image_bundle: dict[str, torch.Tensor],
        current_descriptors: torch.Tensor,
        difference_prompt: torch.Tensor,
    ) -> list[str]:
        current_embed = image_bundle["current_embed"]
        prompt_embed, prompt_mask = self.refine(
            difference_prompt,
            current_descriptors,
            current_embed,
        )
        bos_embed, bos_mask = self._bos(current_embed.shape[0], current_embed.device)
        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds=torch.cat([bos_embed, prompt_embed], dim=1),
                attention_mask=torch.cat([bos_mask, prompt_mask], dim=1),
                **self._generation_kwargs(),
            )
        return self._decode_batch(outputs)

    def _encode_report_descriptors(self, reports: list[str]) -> torch.Tensor:
        device = self.lytim_controller.fact_embeddings.device
        tokens = self.llama_tokenizer(
            reports,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            hidden = self.llama_model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                return_dict=True,
                output_hidden_states=True,
            ).hidden_states[-1]
        return self.text_compressor(hidden.detach())

    def _chexbert_probabilities(self, reports: Iterable[str]) -> torch.Tensor:
        with torch.no_grad():
            probabilities = self.chexbert_metrics.chexbert(list(reports), return_logits=True)
        return probabilities.float()

    def _resolve_training_seeds(
        self,
        samples: dict[str, Any],
        image_bundle: dict[str, torch.Tensor],
    ) -> tuple[list[str], list[str]]:
        source = str(self._arg("lytim_seed_source", "auto")).lower()
        previous_reports = list(samples["prev_text"])
        current_ground_truth = list(samples["curr_text"])
        precomputed_current = samples.get("curr_stage1_text")
        precomputed_previous = samples.get("prev_stage1_text")

        if source not in {"auto", "stage1", "precomputed", "teacher"}:
            raise ValueError(
                "--lytim_seed_source must be one of auto, stage1, precomputed, or teacher."
            )
        if source == "precomputed" and precomputed_current is None:
            raise KeyError(
                "lytim_seed_source=precomputed requires batch key 'curr_stage1_text'."
            )

        if source == "teacher":
            return current_ground_truth, previous_reports
        if source in {"auto", "precomputed"} and precomputed_current is not None:
            current_seed = list(precomputed_current)
        else:
            current_seed = self._generate_stage1_current(image_bundle, previous_reports)

        if source in {"auto", "precomputed"} and precomputed_previous is not None:
            backward_seed = list(precomputed_previous)
        else:
            backward_seed = self._generate_backward_prior(image_bundle, current_seed)
        return current_seed, backward_seed

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(self, samples: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.llama_tokenizer.padding_side = "right"
        previous_reports = list(samples["prev_text"])
        current_reports = list(samples["curr_text"])
        image_bundle = self._encode_longitudinal_images(samples)
        device = image_bundle["current_embed"].device
        batch_size = image_bundle["current_embed"].shape[0]

        image_outputs = self.lytim_controller.encode_images(
            image_bundle["previous_embed"],
            image_bundle["current_embed"],
        )
        previous_gt_probabilities = self._chexbert_probabilities(previous_reports).to(device)
        current_gt_probabilities = self._chexbert_probabilities(current_reports).to(device)
        previous_labels = previous_gt_probabilities.argmax(dim=-1)
        current_labels = current_gt_probabilities.argmax(dim=-1)
        transition_targets, transition_mask = transition_targets_from_labels(
            previous_labels,
            current_labels,
        )

        current_seed, backward_seed = self._resolve_training_seeds(samples, image_bundle)
        current_seed_probabilities = self._chexbert_probabilities(current_seed).to(device)
        backward_seed_probabilities = self._chexbert_probabilities(backward_seed).to(device)
        current_seed_labels = current_seed_probabilities.argmax(dim=-1)
        _, initial_transition_mask = transition_targets_from_labels(
            previous_labels,
            current_seed_labels,
        )

        current_seed_descriptors = self._encode_report_descriptors(current_seed)
        initial_vector, initial_state_logits = self.lytim_controller.encode_report(
            current_seed_descriptors
        )
        initial_backward_logits = self.lytim_controller.predict_backward_state(
            initial_vector,
            image_outputs["progression_vector"],
        )

        image_state_probabilities = image_outputs["image_state_logits"].softmax(dim=-1)
        progression_probabilities = image_outputs["progression_logits"].softmax(dim=-1)
        initial_energy = self.lytim_controller.compute_energy(
            report_probabilities=current_seed_probabilities,
            image_probabilities=image_state_probabilities,
            progression_probabilities=progression_probabilities,
            previous_probabilities=previous_gt_probabilities,
            backward_probabilities=backward_seed_probabilities,
            transition_mask=initial_transition_mask,
        )

        difference_prompt = self.triplet_encoder(
            previous_labels,
            backward_seed_probabilities.argmax(dim=-1),
        )
        difference_prompt, fact_gate = self.lytim_controller.gate_refinement_prompt(
            difference_prompt,
            initial_energy.fact_errors,
            initial_vector,
        )

        bos_embed, bos_mask = self._bos(batch_size, device)
        refine_prompt, refine_mask = self.refine(
            difference_prompt,
            current_seed_descriptors,
            image_bundle["current_embed"],
        )
        target_reports = [report + self.end_sym for report in current_reports]
        target_tokens, target_embeds, targets = self.training_input_generate(
            target_reports,
            refine_mask.shape[1],
            device,
        )
        model_inputs = torch.cat([bos_embed, refine_prompt, target_embeds], dim=1)
        model_mask = torch.cat([bos_mask, refine_mask, target_tokens.attention_mask], dim=1)
        refine_outputs = self.llama_model(
            inputs_embeds=model_inputs,
            attention_mask=model_mask,
            return_dict=True,
            labels=targets,
            output_hidden_states=True,
        )

        target_length = target_tokens.input_ids.shape[1]
        refined_hidden = refine_outputs.hidden_states[-1][:, -target_length:, :]
        refined_descriptors = self.text_compressor(refined_hidden)
        refined_vector, refined_state_logits = self.lytim_controller.encode_report(
            refined_descriptors
        )
        refined_backward_logits = self.lytim_controller.predict_backward_state(
            refined_vector,
            image_outputs["progression_vector"],
        )
        refined_state_probabilities = refined_state_logits.softmax(dim=-1)
        refined_backward_probabilities = refined_backward_logits.softmax(dim=-1)
        refined_energy = self.lytim_controller.compute_energy(
            report_probabilities=refined_state_probabilities,
            image_probabilities=image_state_probabilities,
            progression_probabilities=progression_probabilities,
            previous_probabilities=previous_gt_probabilities,
            backward_probabilities=refined_backward_probabilities,
            transition_mask=transition_mask,
        )

        state_supervision = (
            safe_cross_entropy(image_outputs["image_state_logits"], current_labels)
            + safe_cross_entropy(initial_state_logits, current_seed_labels)
            + safe_cross_entropy(refined_state_logits, current_labels)
        ) / 3.0
        progression_supervision = safe_cross_entropy(
            image_outputs["progression_logits"],
            transition_targets,
        )
        backward_supervision = 0.5 * (
            safe_cross_entropy(initial_backward_logits, previous_labels)
            + safe_cross_entropy(refined_backward_logits, previous_labels)
        )
        supervision_loss = state_supervision + progression_supervision + backward_supervision

        margin = float(self._arg("lytim_margin", 0.02))
        monotonic_loss = F.relu(refined_energy.total - initial_energy.total + margin).mean()

        initial_detached = current_seed_probabilities.detach().clamp_min(1e-6)
        refined_safe = refined_state_probabilities.clamp_min(1e-6)
        fact_kl = (
            initial_detached * (initial_detached.log() - refined_safe.log())
        ).sum(dim=-1)
        consistent_facts = (
            current_seed_labels.eq(current_labels)
            & image_outputs["image_state_logits"].argmax(dim=-1).eq(current_labels)
        )
        keep_loss = masked_mean(fact_kl, consistent_facts, dim=-1).mean()

        descriptor_change = 1.0 - F.cosine_similarity(initial_vector, refined_vector, dim=-1)
        stop_logits = self.lytim_controller.stop_logits(
            initial_energy.total.detach(),
            refined_energy.total.detach(),
            descriptor_change.detach(),
            refined_energy.fact_errors.detach(),
        )
        improvement = initial_energy.total.detach() - refined_energy.total.detach()
        stop_target = (
            refined_energy.total.detach().le(float(self._arg("lytim_stop_energy", 0.15)))
            | improvement.lt(float(self._arg("lytim_accept_epsilon", 0.01)))
        ).float()
        stop_loss = F.binary_cross_entropy_with_logits(stop_logits, stop_target)
        gate_sparsity_loss = fact_gate.mean()

        loss = (
            float(self._arg("lytim_lm_weight", 1.0)) * refine_outputs.loss
            + float(self._arg("lytim_supervision_weight", 0.2)) * supervision_loss
            + float(self._arg("lytim_monotonic_weight", 0.1)) * monotonic_loss
            + float(self._arg("lytim_keep_weight", 0.05)) * keep_loss
            + float(self._arg("lytim_stop_weight", 0.02)) * stop_loss
            + float(self._arg("lytim_gate_weight", 0.001)) * gate_sparsity_loss
        )

        return {
            "loss": loss,
            "lm_loss": refine_outputs.loss.detach(),
            "clinical_supervision_loss": supervision_loss.detach(),
            "monotonic_loss": monotonic_loss.detach(),
            "keep_loss": keep_loss.detach(),
            "stop_loss": stop_loss.detach(),
            "gate_sparsity": gate_sparsity_loss.detach(),
            "energy_initial": initial_energy.total.mean().detach(),
            "energy_refined": refined_energy.total.mean().detach(),
            "energy_drop": (initial_energy.total - refined_energy.total).mean().detach(),
        }

    # ------------------------------------------------------------------
    # Monotonic inference
    # ------------------------------------------------------------------
    @staticmethod
    def _select_tensor(
        current: torch.Tensor,
        candidate: torch.Tensor,
        accept: torch.Tensor,
    ) -> torch.Tensor:
        view_shape = [accept.shape[0]] + [1] * (current.ndim - 1)
        return torch.where(accept.view(*view_shape), candidate, current)

    def _select_breakdown(
        self,
        current: EnergyBreakdown,
        candidate: EnergyBreakdown,
        accept: torch.Tensor,
    ) -> EnergyBreakdown:
        return EnergyBreakdown(
            total=self._select_tensor(current.total, candidate.total, accept),
            state=self._select_tensor(current.state, candidate.state, accept),
            change=self._select_tensor(current.change, candidate.change, accept),
            backward=self._select_tensor(current.backward, candidate.backward, accept),
            fact_errors=self._select_tensor(current.fact_errors, candidate.fact_errors, accept),
        )

    def _score_generated_reports(
        self,
        reports: list[str],
        backward_reports: list[str],
        descriptors: torch.Tensor,
        image_outputs: dict[str, torch.Tensor],
        previous_gt_probabilities: torch.Tensor,
    ) -> dict[str, Any]:
        device = descriptors.device
        report_vector, learned_state_logits = self.lytim_controller.encode_report(descriptors)
        learned_backward_logits = self.lytim_controller.predict_backward_state(
            report_vector,
            image_outputs["progression_vector"],
        )
        report_probabilities = self._chexbert_probabilities(reports).to(device)
        generated_backward_probabilities = self._chexbert_probabilities(backward_reports).to(device)
        learned_backward_probabilities = learned_backward_logits.softmax(dim=-1)
        generated_weight = float(self._arg("lytim_generated_backward_weight", 0.5))
        generated_weight = min(max(generated_weight, 0.0), 1.0)
        backward_probabilities = (
            generated_weight * generated_backward_probabilities
            + (1.0 - generated_weight) * learned_backward_probabilities
        )

        _, transition_mask = transition_targets_from_labels(
            previous_gt_probabilities.argmax(dim=-1),
            report_probabilities.argmax(dim=-1),
        )
        energy = self.lytim_controller.compute_energy(
            report_probabilities=report_probabilities,
            image_probabilities=image_outputs["image_state_logits"].softmax(dim=-1),
            progression_probabilities=image_outputs["progression_logits"].softmax(dim=-1),
            previous_probabilities=previous_gt_probabilities,
            backward_probabilities=backward_probabilities,
            transition_mask=transition_mask,
        )
        return {
            "reports": reports,
            "backward_reports": backward_reports,
            "descriptors": descriptors,
            "report_vector": report_vector,
            "generated_backward_probabilities": generated_backward_probabilities,
            "energy": energy,
        }

    def _monotonic_generate(self, samples: dict[str, Any]) -> tuple[list[str], dict[str, torch.Tensor]]:
        self.llama_tokenizer.padding_side = "right"
        previous_reports = list(samples["prev_text"])
        image_bundle = self._encode_longitudinal_images(samples)
        image_outputs = self.lytim_controller.encode_images(
            image_bundle["previous_embed"],
            image_bundle["current_embed"],
        )
        device = image_bundle["current_embed"].device
        previous_gt_probabilities = self._chexbert_probabilities(previous_reports).to(device)
        previous_labels = previous_gt_probabilities.argmax(dim=-1)

        current_reports = self._generate_stage1_current(image_bundle, previous_reports)
        backward_reports = self._generate_backward_prior(image_bundle, current_reports)
        descriptors = self._encode_report_descriptors(current_reports)
        current = self._score_generated_reports(
            current_reports,
            backward_reports,
            descriptors,
            image_outputs,
            previous_gt_probabilities,
        )

        initial_energy = current["energy"].total.clone()
        batch_size = initial_energy.shape[0]
        accepted_steps = torch.zeros(batch_size, dtype=torch.long, device=device)
        rolled_back = torch.zeros(batch_size, dtype=torch.bool, device=device)
        stopped_by_head = torch.zeros(batch_size, dtype=torch.bool, device=device)
        active = current["energy"].total.gt(float(self._arg("lytim_stop_energy", 0.15)))

        max_iterations = int(self.hparams.max_iteration)
        acceptance_epsilon = float(self._arg("lytim_accept_epsilon", 0.01))
        stop_threshold = float(self._arg("lytim_stop_threshold", 0.5))
        minimum_iterations = int(self._arg("lytim_min_iterations", 0))

        for iteration in range(max_iterations):
            if not bool(active.any()):
                break

            backward_labels = current["generated_backward_probabilities"].argmax(dim=-1)
            difference_prompt = self.triplet_encoder(previous_labels, backward_labels)
            difference_prompt, _ = self.lytim_controller.gate_refinement_prompt(
                difference_prompt,
                current["energy"].fact_errors,
                current["report_vector"],
            )
            candidate_reports = self._generate_refined_current(
                image_bundle,
                current["descriptors"],
                difference_prompt,
            )
            candidate_backward_reports = self._generate_backward_prior(
                image_bundle,
                candidate_reports,
            )
            candidate_descriptors = self._encode_report_descriptors(candidate_reports)
            candidate = self._score_generated_reports(
                candidate_reports,
                candidate_backward_reports,
                candidate_descriptors,
                image_outputs,
                previous_gt_probabilities,
            )

            descriptor_change = 1.0 - F.cosine_similarity(
                current["report_vector"],
                candidate["report_vector"],
                dim=-1,
            )
            stop_probability = torch.sigmoid(
                self.lytim_controller.stop_logits(
                    current["energy"].total,
                    candidate["energy"].total,
                    descriptor_change,
                    candidate["energy"].fact_errors,
                )
            )
            accept = monotonic_acceptance_mask(
                current["energy"].total,
                candidate["energy"].total,
                acceptance_epsilon,
                active=active,
            )
            rolled_back |= active & ~accept
            accepted_steps += accept.long()

            current["reports"] = [
                candidate_reports[index] if bool(accept[index]) else current["reports"][index]
                for index in range(batch_size)
            ]
            current["backward_reports"] = [
                candidate_backward_reports[index]
                if bool(accept[index])
                else current["backward_reports"][index]
                for index in range(batch_size)
            ]
            for key in (
                "descriptors",
                "report_vector",
                "generated_backward_probabilities",
            ):
                current[key] = self._select_tensor(current[key], candidate[key], accept)
            current["energy"] = self._select_breakdown(
                current["energy"],
                candidate["energy"],
                accept,
            )

            stop_after_accept = accept & (
                stop_probability.ge(stop_threshold)
                | candidate["energy"].total.le(float(self._arg("lytim_stop_energy", 0.15)))
            )
            if iteration + 1 < minimum_iterations:
                stop_after_accept = torch.zeros_like(stop_after_accept)
            stopped_by_head |= stop_after_accept
            active = active & accept & ~stop_after_accept

        diagnostics = {
            "accepted_steps": accepted_steps,
            "initial_energy": initial_energy,
            "final_energy": current["energy"].total,
            "energy_drop": initial_energy - current["energy"].total,
            "rolled_back": rolled_back.float(),
            "stopped_by_head": stopped_by_head.float(),
        }
        return current["reports"], diagnostics

    def _evaluation_step(self, samples: dict[str, Any], split: str) -> None:
        reports, diagnostics = self._monotonic_generate(samples)
        references = self.set_report_length(list(samples["curr_text"]))
        outputs = {
            "ref": references,
            "id": list(samples["id"]),
            "hypo": reports,
            **{key: value.detach().cpu().tolist() for key, value in diagnostics.items()},
        }
        if split == "val":
            self.val_step_outputs.append(outputs)
        elif split == "test":
            self.test_step_outputs.append(outputs)
        else:
            raise ValueError(f"Unknown evaluation split: {split}")

    def validation_step(self, samples: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step(samples, "val")

    def test_step(self, samples: dict[str, Any], batch_idx: int) -> None:
        del batch_idx
        self._evaluation_step(samples, "test")

    def _log_refinement_diagnostics(self, outputs: list[dict[str, Any]], prefix: str) -> None:
        if not outputs:
            return
        metrics: dict[str, torch.Tensor] = {}
        for key in (
            "accepted_steps",
            "initial_energy",
            "final_energy",
            "energy_drop",
            "rolled_back",
            "stopped_by_head",
        ):
            values = [value for output in outputs for value in output.get(key, [])]
            if values:
                metrics[f"{prefix}_{key}"] = torch.tensor(
                    values,
                    dtype=torch.float32,
                    device=self.device,
                ).mean()
        if metrics:
            self.log_dict(metrics, sync_dist=True, logger=True)

    def on_validation_epoch_end(self) -> None:
        self._log_refinement_diagnostics(self.val_step_outputs, "lytim_val")
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        self._log_refinement_diagnostics(self.test_step_outputs, "lytim_test")
        super().on_test_epoch_end()

    def configure_optimizers(self):
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise RuntimeError("LyTIM has no trainable parameters.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.hparams.learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
