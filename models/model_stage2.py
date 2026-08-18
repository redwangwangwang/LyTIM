import json
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import SwinModel
import torch.nn.functional as F

from evalcap.bleu.bleu import Bleu
from evalcap.cider.cider import Cider
from evalcap.rouge.rouge import Rouge
from evalcap.meteor.meteor import Meteor
from evalcap.ce_metrics.metrics_clinical import CheXbertMetrics
from models.utils.XCLIP import XCLIPEncoder
from models.utils.triplet_encoder import TripletEncoder
from models.utils.attention_text_compressor import AttentionCompressor

DISEASE = ['enlarged cardiomediastinum',
           'cardiomegaly',
           'lung opacity',
           'lung lesion',
           'edema',
           'consolidation',
           'pneumonia',
           'atelectasis',
           'pneumothorax',
           'pleural effusion',
           'pleural other',
           'fracture',
           'support devices',
           'no finding']
CONDITIONS = {0: 'not mentioned', 1: 'positive', 2: 'negative', 3: 'uncertain'}


class LongitudinalR2GenGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # ----------------------------Loading Vision Encoder----------------------------
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        print(f'Loading vision encoder:{args.vision_model} -- Done')
        # ----------------------------Loading Vision Encoder Done----------------------------

        # ----------------------------loading LLaMA----------------------------
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['<Image>', '<Progression>', '<Diff>', '<Report>']})
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        self.image_token_id = self.llama_tokenizer.convert_tokens_to_ids("<Image>")
        self.progression_token_id = self.llama_tokenizer.convert_tokens_to_ids("<Progression>")
        self.diff_token_id = self.llama_tokenizer.convert_tokens_to_ids("<Diff>")
        self.report_token_id = self.llama_tokenizer.convert_tokens_to_ids("<Report>")

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
        # ----------------------------Loading LLaMA Done----------------------------

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'You are a radiologist. Generate a comprehensive and detailed diagnosis report for the chest xray image based on report of prior image and change between prior and current image.'
        self.prompt_prior = 'You are a radiologist. Here is a chest X-ray image, along with the changes compared to subsequent image. Please generate a comprehensive and detailed diagnosis report for this image.'
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        # ----------------------------Loading Progression Modeling----------------------------
        self.perception_frame = nn.Parameter(torch.randn(2, 3, 224, 224), requires_grad=True)
        self.video_encoder = XCLIPEncoder(num_frames=4)
        self.video_linear = nn.Linear(self.video_encoder.hidden_size, self.llama_model.config.hidden_size)
        self.video_layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.perception_agg = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        # ----------------------------Loading Progression Modeling Done----------------------------

        self.chexbert_metrics = CheXbertMetrics(
            './pretrain_weights/chexbert.pth',
            args.batch_size,
        )

        # ----------------------------Stage 2----------------------------
        self.triplet_encoder = TripletEncoder(llama_embed_dim=self.llama_model.config.hidden_size, num_prompt_tokens=16)
        self.text_compressor = AttentionCompressor(self.llama_model.config.hidden_size)
        self.compare_layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        print("Loading Stage 2 Module Done")

        # ----------------------------Freeze Stage1 Model----------------------------
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load delta file from {args.delta_file}')

        for name, param in self.named_parameters():
            if 'triplet_encoder' in name or 'text_compressor' in name or 'compare_layer_norm' in name:
                pass
            else:
                param.requires_grad = False
        print("Freezing Parameters Done")

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)

        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.llama_proj(image_embeds)
        return inputs_llama

    def set_report_length(self, report):
        tokens = self.llama_tokenizer(
            report,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        report = [self.decode(i) for i in tokens.input_ids]
        return report

    def prompt_wrap(self, image_embed, prog_embed, context, timepoint='curr'):
        batch_size = image_embed.shape[0]
        device = image_embed.device
        instruction = self.prompt if timepoint=='curr' else self.prior_prompt

        prompt = f'User: Progression: <Progression>. Context report: <Context>. Image: <Image>. {instruction} \n Assistant:'
        context_tokens = self.llama_tokenizer(
            context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        context = self.llama_tokenizer.batch_decode(context_tokens.input_ids,
                                                    add_special_tokens=False,
                                                    skip_special_tokens=True)
        prompt = [prompt.replace('<Context>', context[i]) for i in range(len(context))]
        tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=200,
            add_special_tokens=False
        )
        embeds_temp = self.embed_tokens(tokens.input_ids.to(device))

        image_token_pos = (tokens.input_ids == self.image_token_id).nonzero(as_tuple=False)[:,1]
        prog_token_pos = (tokens.input_ids == self.progression_token_id).nonzero(as_tuple=False)[:,1]
        assert image_token_pos.shape[0] == batch_size
        assert prog_token_pos.shape[0] == batch_size

        embeds = torch.zeros(batch_size,
                             embeds_temp.shape[1] + image_embed.shape[1] + prog_embed.shape[1] - 2,
                             embeds_temp.shape[2]).to(dtype=embeds_temp.dtype, device=device)
        for i in range(batch_size):
            embeds[i] = torch.cat(tensors=[embeds_temp[i, :prog_token_pos[i], :],
                                           prog_embed[i],
                                           embeds_temp[i, prog_token_pos[i] + 1:image_token_pos[i], :],
                                           image_embed[i],
                                           embeds_temp[i, image_token_pos[i] + 1:, :]], dim=0)
        image_attn = torch.ones(image_embed.size()[:-1], dtype=tokens.attention_mask.dtype)
        prog_attn = torch.ones(prog_embed.size()[:-1], dtype=tokens.attention_mask.dtype)
        attns = torch.cat([image_attn, prog_attn, tokens.attention_mask[:, 2:]], dim=1).to(device)
        return embeds, attns

    def training_input_generate(self, text, prompt_len, device):
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(device)
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        batch_size = to_regress_embeds.shape[0]

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = torch.ones([batch_size, prompt_len + 1],
                                   dtype=torch.long,
                                   device=device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        return to_regress_tokens, to_regress_embeds, targets

    def refine(self, difference_prompt_embed, curr_report_embed, image_embed):
        batch_size = difference_prompt_embed.shape[0]
        device = image_embed.device
        prompt = (f'User: You are a radiologist. '
                  f'You have already written reports for the current and previous chest X-ray images of a patient, '
                  f'but there might be some issues in those reports. '
                  f'Current image: <Image>. '
                  f'Your report: <Report> '
                  f'The following pathology was not consistent with the actual report in the generation of your previous report: <Diff>'
                  f'Please pay special attention to these error and regenerate an accurate report. Assistant:')

        prompt = [prompt for _ in range(batch_size)]
        tokens = self.llama_tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=300,
            add_special_tokens=False
        )
        embeds_temp = self.embed_tokens(tokens.input_ids.to(device))

        image_token_pos = (tokens.input_ids == self.image_token_id).nonzero(as_tuple=False)[:, 1]
        diff_token_pos = (tokens.input_ids == self.diff_token_id).nonzero(as_tuple=False)[:, 1]
        report_token_pos = (tokens.input_ids == self.report_token_id).nonzero(as_tuple=False)[:, 1]
        assert image_token_pos.shape[0] == batch_size
        assert diff_token_pos.shape[0] == batch_size
        assert report_token_pos.shape[0] == batch_size

        embeds = torch.zeros(batch_size,
                             embeds_temp.shape[1] + image_embed.shape[1] + difference_prompt_embed.shape[1] +
                             curr_report_embed.shape[1] - 3,
                             embeds_temp.shape[2]).to(dtype=embeds_temp.dtype, device=device)
        for i in range(batch_size):
            embeds[i] = torch.cat(tensors=[embeds_temp[i, :image_token_pos[i], :],
                                           image_embed[i],
                                           embeds_temp[i, image_token_pos[i] + 1:report_token_pos[i], :],
                                           curr_report_embed[i],
                                           embeds_temp[i, report_token_pos[i] + 1:diff_token_pos[i], :],
                                           difference_prompt_embed[i],
                                           embeds_temp[i, diff_token_pos[i] + 1:]], dim=0)
        image_attn = torch.ones(image_embed.size()[:-1], dtype=tokens.attention_mask.dtype)
        diff_attn = torch.ones(difference_prompt_embed.size()[:-1], dtype=tokens.attention_mask.dtype)
        report_attn = torch.ones(curr_report_embed.size()[:-1], dtype=tokens.attention_mask.dtype)
        attns = torch.cat([image_attn, diff_attn, report_attn, tokens.attention_mask[:, 3:]], dim=1).to(device)

        return embeds, attns

    def forward(self, samples):
        self.llama_tokenizer.padding_side = "right"
        image2 = samples["curr_image"]
        prev_text = samples["prev_text"]
        curr_text = samples["curr_text"]
        prev_stage1_text = samples["prev_stage1_text"]
        curr_stage1_text = samples["curr_stage1_text"]

        img_embeds2 = self.encode_img(image2)
        img_embeds2 = self.layer_norm(img_embeds2)
        device = img_embeds2.device
        batch_size = img_embeds2.shape[0]

        # report compress
        curr_stage1_tokens = self.llama_tokenizer(
            curr_stage1_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(device)
        curr_stage1_embeds = self.llama_model(
            input_ids=curr_stage1_tokens.input_ids,
            attention_mask=curr_stage1_tokens.attention_mask,
            return_dict=True,
            output_hidden_states=True,
        ).hidden_states[-1][:, -self.hparams.max_length:, :]
        curr_report_descriptors = self.text_compressor(curr_stage1_embeds)

        # error triplet encode
        prev_gt_text = self.set_report_length(prev_text)
        hypo_label = self.chexbert_metrics.chexbert(list(prev_stage1_text))
        ref_label = self.chexbert_metrics.chexbert(list(prev_gt_text))
        diff_embed = self.triplet_encoder(ref_label, hypo_label)

        # begin of sentence token
        bos = torch.ones([batch_size, 1],
                         dtype=curr_stage1_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_attns = curr_stage1_tokens.attention_mask[:, :1]

        refine_prompt_embeds, refine_prompt_attns = self.refine(diff_embed, curr_report_descriptors, img_embeds2)
        refine_to_regress_tokens, refine_to_regress_embeds, refine_targets = self.training_input_generate(curr_text, refine_prompt_attns.shape[1], device)
        refine_prompt_embeds = torch.cat([bos_embeds, refine_prompt_embeds, refine_to_regress_embeds], dim=1)
        refine_prompt_attns = torch.cat([bos_attns, refine_prompt_attns, refine_to_regress_tokens.attention_mask], dim=1)
        refine_outputs = self.llama_model(
            inputs_embeds=refine_prompt_embeds,
            attention_mask=refine_prompt_attns,
            return_dict=True,
            labels=refine_targets,
            output_hidden_states=True,
        )

        # refinement loss
        curr_gt_tokens = self.llama_tokenizer(
            curr_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(device)
        curr_gt_embeds = self.llama_model(
            input_ids=curr_gt_tokens.input_ids,
            attention_mask=curr_gt_tokens.attention_mask,
            return_dict=True,
            output_hidden_states=True,
        ).hidden_states[-1][:, -self.hparams.max_length:, :]

        curr_gt_embeds = self.compare_layer_norm(torch.max(curr_gt_embeds, dim=1)[0])
        curr_first_pred_embeds = self.compare_layer_norm(torch.max(curr_stage1_embeds, dim=1)[0])
        curr_second_pred_embeds = self.compare_layer_norm(torch.max(refine_outputs.hidden_states[-1][:, -self.hparams.max_length:, :], dim=1)[0])

        sim_first_pred = F.cosine_similarity(curr_gt_embeds, curr_first_pred_embeds)
        sim_second_pred = F.cosine_similarity(curr_gt_embeds, curr_second_pred_embeds)

        beta = 5
        distance_loss = (-torch.log(torch.sigmoid(beta * (sim_second_pred - sim_first_pred)))).mean()

        loss = refine_outputs.loss + 0.1 * distance_loss
        return {"loss": loss, "distance_loss": distance_loss, "refine_output_loss": refine_outputs.loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res, val_score):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "state_dict": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
            "pytorch-lightning_version": pl.__version__,
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:.3f}_cider{:.3f}.pth".format(current_epoch, global_step,
                                                                          eval_res['Bleu_4'],
                                                                          eval_res['CIDEr']),
        )

        self.print(
            "[Epoch {}] Saving checkpoint at step {} to {}.".format(self.trainer.current_epoch, global_step, save_to))
        torch.save(save_obj, save_to)

        if val_score > self.val_score and self.trainer.global_step != 0:
            save_best = os.path.join(self.hparams.savedmodel_path, 'checkpoints', 'best.pth')
            torch.save(save_obj, save_best)
            self.val_score = val_score
            self.print("[Epoch {}] Saving best checkpoint at step {}.".format(self.trainer.current_epoch, global_step))
        self.print("")

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['curr_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        # ------------------------------------------STAGE 1------------------------------------------
        image1 = samples["prev_image"]
        image2 = samples["curr_image"]
        prev_text = samples["prev_text"]
        curr_text = samples["curr_text"]

        # pathology encoding
        img_embeds1 = self.encode_img(image1)
        img_embeds1 = self.layer_norm(img_embeds1)
        img_embeds2 = self.encode_img(image2)
        img_embeds2 = self.layer_norm(img_embeds2)
        batch_size = img_embeds1.shape[0]
        device = img_embeds1.device

        # progression encoding
        p_frame = self.perception_frame.expand(batch_size, -1, -1, -1, -1)
        video = torch.stack([image1[0], p_frame[:, 0], p_frame[:, 1], image2[0]], dim=1)
        perception_embed = self.video_encoder(video)[:, 1:3, :, :]
        alpha = torch.sigmoid(self.perception_agg(perception_embed).squeeze(1))
        perception_embed = alpha * perception_embed[:, 0, :, :] + (1 - alpha) * perception_embed[:, 1, :, :]
        perception_embed = self.video_layer_norm(self.video_linear(perception_embed))

        # begin of sentence token
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_attns = to_regress_tokens.attention_masks[:, :1]

        # generate current report
        curr_prompt_embeds, curr_prompt_attns = self.prompt_wrap(img_embeds2, perception_embed, prev_text, timepoint='curr')
        curr_inputs_embeds = torch.cat([bos_embeds, curr_prompt_embeds], dim=1)
        curr_attention_mask = torch.cat([bos_attns, curr_prompt_attns], dim=1)
        curr_outputs = self.llama_model.generate(
            inputs_embeds=curr_inputs_embeds,
            attention_mask=curr_attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=None,
            top_p=None,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        curr_output_report = [self.decode(i) for i in curr_outputs.sequences]

        prev_gt_report = self.set_report_length(prev_text)
        curr_gt_report = self.set_report_length(curr_text)

        # ------------------------------------------STAGE 2------------------------------------------
        for i in range(self.hparams.max_iteration):
            # current report compress
            last_layer_hiddens = [step_hidden[-1] for step_hidden in curr_outputs.hidden_states]
            final_hidden_states = torch.cat(last_layer_hiddens, dim=1)
            curr_output_embed = final_hidden_states[::self.hparams.beam_size][:, -self.hparams.max_length:, :]
            curr_output_embed = self.text_compressor(curr_output_embed)

            # generate prior report
            prev_prompt_embeds, prev_prompt_attns = self.prompt_wrap(img_embeds1, perception_embed, curr_output_report, timepoint='prior')
            prev_inputs_embeds = torch.cat([bos_embeds, prev_prompt_embeds], dim=1)
            prev_attention_mask = torch.cat([bos_attns, prev_prompt_attns], dim=1)
            prev_outputs = self.llama_model.generate(
                inputs_embeds=prev_inputs_embeds,
                attention_mask=prev_attention_mask,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=None,
                top_p=None,
                use_cache=True
            )
            prev_pred_report = [self.decode(i) for i in prev_outputs]
            hypo_label = self.chexbert_metrics.chexbert(list(prev_pred_report))
            ref_label = self.chexbert_metrics.chexbert(list(prev_gt_report))
            diff_prompt_embed = self.triplet_encoder(ref_label, hypo_label)

            refine_prompt_embeds, refine_prompt_attns = self.refine(diff_prompt_embed, curr_output_embed, img_embeds2)
            refine_prompt_embeds = torch.cat([bos_embeds, refine_prompt_embeds], dim=1)
            refine_prompt_attns = torch.cat([bos_attns, refine_prompt_attns], dim=1)
            curr_outputs = self.llama_model.generate(
                inputs_embeds=refine_prompt_embeds,
                attention_mask=refine_prompt_attns,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            curr_output_report = [self.decode(i) for i in curr_outputs.sequences]

        outputs = {"ref": curr_gt_report, "id": samples["id"], "hypo": curr_output_report}
        self.val_step_outputs.append(outputs)

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, ids, hypo = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            ids.extend(i['id'])
            hypo.extend(i['hypo'])

        ref_1 = {k: [v] for k, v in zip(ids, ref)}
        hypo_1 = {k: [v] for k, v in zip(ids, hypo)}
        ref = [ref_1[id][0] for id in sorted(ref_1.keys())]
        hypo = [hypo_1[id][0] for id in sorted(hypo_1.keys())]
        eval_res = self.score(ref=ref_1, hypo=hypo_1)
        eval_ce, _ = self.chexbert_metrics.compute(ref, hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        self.log_dict(eval_ce, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        with open(os.path.join(result_folder, 'refs.json'), 'w') as f2:
            json.dump(ref_1, f2)
        with open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w') as f3:
            json.dump(hypo_1, f3)

        self.print('[Epoch {}] {}'.format(self.trainer.current_epoch, {k: round(v, 5) for k, v in eval_res.items()}))
        self.print('[Epoch {}] {}'.format(self.trainer.current_epoch, {k: round(v, 5) for k, v in eval_ce.items()}))

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            self.save_checkpoint(eval_res, val_score)
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['curr_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        # ------------------------------------------STAGE 1------------------------------------------
        image1 = samples["prev_image"]
        image2 = samples["curr_image"]
        prev_text = samples["prev_text"]
        curr_text = samples["curr_text"]
        prog_text = samples["progressions"]

        # pathology encoding
        img_embeds1 = self.encode_img(image1)
        img_embeds1 = self.layer_norm(img_embeds1)
        img_embeds2 = self.encode_img(image2)
        img_embeds2 = self.layer_norm(img_embeds2)
        batch_size = img_embeds1.shape[0]
        device = img_embeds1.device

        # progression encoding
        p_frame = self.perception_frame.expand(batch_size, -1, -1, -1, -1)
        video = torch.stack([image1[0], p_frame[:, 0], p_frame[:, 1], image2[0]], dim=1)
        perception_embed = self.video_encoder(video)[:, 1:3, :, :]
        alpha = torch.sigmoid(self.perception_agg(perception_embed).squeeze(1))
        perception_embed = alpha * perception_embed[:, 0, :, :] + (1 - alpha) * perception_embed[:, 1, :, :]
        perception_embed = self.video_layer_norm(self.video_linear(perception_embed))

        # begin of sentence token
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_attns = to_regress_tokens.attention_masks[:, :1]

        # generate current report
        curr_prompt_embeds, curr_prompt_attns = self.prompt_wrap(img_embeds2, perception_embed, prev_text, timepoint='curr')
        curr_inputs_embeds = torch.cat([bos_embeds, curr_prompt_embeds], dim=1)
        curr_attention_mask = torch.cat([bos_attns, curr_prompt_attns], dim=1)
        curr_outputs = self.llama_model.generate(
            inputs_embeds=curr_inputs_embeds,
            attention_mask=curr_attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=None,
            top_p=None,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        curr_output_report = [self.decode(i) for i in curr_outputs.sequences]

        prev_gt_report = self.set_report_length(prev_text)
        curr_gt_report = self.set_report_length(curr_text)

        # ------------------------------------------STAGE 2------------------------------------------
        for i in range(self.hparams.max_iteration):
            # current report compress
            last_layer_hiddens = [step_hidden[-1] for step_hidden in curr_outputs.hidden_states]
            final_hidden_states = torch.cat(last_layer_hiddens, dim=1)
            curr_output_embed = final_hidden_states[::self.hparams.beam_size][:, -self.hparams.max_length:, :]
            curr_output_embed = self.text_compressor(curr_output_embed)

            # generate prior report
            prev_prompt_embeds, prev_prompt_attns = self.prompt_wrap(img_embeds1, perception_embed, curr_output_report, timepoint='prior')
            prev_inputs_embeds = torch.cat([bos_embeds, prev_prompt_embeds], dim=1)
            prev_attention_mask = torch.cat([bos_attns, prev_prompt_attns], dim=1)
            prev_outputs = self.llama_model.generate(
                inputs_embeds=prev_inputs_embeds,
                attention_mask=prev_attention_mask,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=None,
                top_p=None,
                use_cache=True
            )
            prev_pred_report = [self.decode(i) for i in prev_outputs]
            hypo_label = self.chexbert_metrics.chexbert(list(prev_pred_report))
            ref_label = self.chexbert_metrics.chexbert(list(prev_gt_report))
            diff_prompt_embed = self.triplet_encoder(ref_label, hypo_label)
            refine_prompt_embeds, refine_prompt_attns = self.refine(diff_prompt_embed, curr_output_embed, img_embeds2)

            refine_prompt_embeds = torch.cat([bos_embeds, refine_prompt_embeds], dim=1)
            refine_prompt_attns = torch.cat([bos_attns, refine_prompt_attns], dim=1)
            curr_outputs = self.llama_model.generate(
                inputs_embeds=refine_prompt_embeds,
                attention_mask=refine_prompt_attns,
                pad_token_id=self.llama_tokenizer.pad_token_id,
                num_beams=self.hparams.beam_size,
                do_sample=self.hparams.do_sample,
                min_new_tokens=self.hparams.min_new_tokens,
                max_new_tokens=self.hparams.max_new_tokens,
                repetition_penalty=self.hparams.repetition_penalty,
                length_penalty=self.hparams.length_penalty,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                output_hidden_states=True
            )
            curr_output_report = [self.decode(i) for i in curr_outputs.logits]

        outputs = {"ref": curr_gt_report, "id": samples["id"], "hypo": curr_output_report}
        self.test_step_outputs.append(outputs)

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, ids, hypo = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            ids.extend(i['id'])
            hypo.extend(i['hypo'])

        ref_1 = {k: [v] for k, v in zip(ids, ref)}
        hypo_1 = {k: [v] for k, v in zip(ids, hypo)}
        ref = [ref_1[id][0] for id in sorted(ref_1.keys())]
        hypo = [hypo_1[id][0] for id in sorted(hypo_1.keys())]

        eval_res = self.score(ref=ref_1, hypo=hypo_1)
        eval_ce, _ = self.chexbert_metrics.compute(ref, hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        self.log_dict(eval_ce, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        with open(os.path.join(result_folder, f'test_refs_localrank{self.local_rank}.json'), 'w') as f2:
            json.dump(ref_1, f2)
        with open(os.path.join(result_folder, f'test_result_localrank{self.local_rank}.json'), 'w') as f3:
            json.dump(hypo_1, f3)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs,
                                                               eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()