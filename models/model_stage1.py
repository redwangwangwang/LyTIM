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
from models.utils.XCLIP_text import XCLIPTextEncoder

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
CONDITIONS = {0: 'not mentioned',
              1: 'positive',
              2: 'negative',
              3: 'uncertain'}

class LongitudinalR2GenGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # ----------------------------Loading Vision Encoder----------------------------
        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')
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
        self.llama_tokenizer.add_special_tokens({'additional_special_tokens': ['<Image>', '<Progression>']})
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        self.image_token_id = self.llama_tokenizer.convert_tokens_to_ids("<Image>")
        self.progression_token_id = self.llama_tokenizer.convert_tokens_to_ids("<Progression>")
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
        self.prompt = 'You are a radiologist. Generate a comprehensive and detailed diagnosis report for the chest xray image based on report of prior image and progression between prior and current image.'
        self.prompt_prior = 'You are a radiologist. Here is a chest X-ray image, along with the progressions compared to subsequent image. Please generate a comprehensive and detailed diagnosis report for this image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        self.text_encoder = XCLIPTextEncoder()
        self.pathology_align_proj = nn.Linear(self.text_encoder.text_model.config.hidden_size, self.llama_model.config.hidden_size)

        # ----------------------------Loading Progression Modeling----------------------------
        self.perception_frame = nn.Parameter(torch.randn(2, 3, 224, 224), requires_grad=True)
        self.video_encoder = XCLIPEncoder(num_frames=4)
        self.video_linear = nn.Linear(self.video_encoder.hidden_size, self.llama_model.config.hidden_size)
        self.video_layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.perception_agg = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.prog_align_proj = nn.Linear(self.text_encoder.text_model.config.hidden_size, self.llama_model.config.hidden_size)
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        # ----------------------------Loading Progression Modeling Done----------------------------

        self.chexbert_metrics = CheXbertMetrics(
            './pretrain_weights/chexbert.pth',
            args.batch_size,
        )
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))[
                'model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

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

    def progression_loss(self, prog_text, perception_embed):
        device = perception_embed.device
        batch_size = perception_embed.shape[0]
        prog_text_features, logit_scale = self.text_encoder(prog_text, device=device)
        prog_text_features = self.prog_align_proj(prog_text_features)
        prog_visual_features = perception_embed[:, 0, :]
        prog_visual_features = prog_visual_features / prog_visual_features.norm(dim=1, keepdim=True)
        prog_text_features = prog_text_features / prog_text_features.norm(dim=1, keepdim=True)
        logits_per_image = logit_scale * prog_visual_features @ prog_text_features.t()

        labels = torch.arange(batch_size, device=device)
        prog_loss = F.cross_entropy(logits_per_image, labels)

        return prog_loss

    def pathology_loss(self, visual_features, reports):
        ce_model = self.chexbert_metrics.chexbert
        labels = ce_model(list(reports))[:, :-1]
        batch_size = len(reports)
        disease_text_list = []
        for batch_idx in range(batch_size):
            description_list = []
            for disease_idx, disease in enumerate(DISEASE):
                if disease_idx == len(DISEASE) - 1:
                    continue
                label = labels[batch_idx][disease_idx].item()
                description_list.append(f'{disease} is {CONDITIONS[label]}.')
            disease_text_list.append(' '.join(description_list))

        text_features, logit_scale = self.text_encoder(disease_text_list, visual_features.device)
        text_features = self.pathology_align_proj(text_features)
        visual_features = visual_features[:, 0, :]
        visual_features = visual_features / visual_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = logit_scale * visual_features @ text_features.t()

        labels = torch.arange(visual_features.shape[0], device=visual_features.device)
        pathology_loss = F.cross_entropy(logits_per_image, labels)

        return pathology_loss

    def forward(self, samples):
        self.llama_tokenizer.padding_side = "right"
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
        device = img_embeds1.device
        batch_size = img_embeds1.shape[0]

        # progression encoding
        p_frame = self.perception_frame.expand(batch_size, -1, -1, -1, -1)
        video = torch.stack([image1[0], p_frame[:, 0], p_frame[:, 1], image2[0]], dim=1)
        perception_embed = self.video_encoder(video)[:, 1:3, :, :]
        alpha = torch.sigmoid(self.perception_agg(perception_embed).squeeze(1))
        perception_embed = alpha * perception_embed[:, 0, :, :] + (1 - alpha) * perception_embed[:, 1, :, :]
        perception_embed = self.video_layer_norm(self.video_linear(perception_embed))

        # prompt wrapping
        prev_prompt_embeds, prev_prompt_attns = self.prompt_wrap(img_embeds1, perception_embed, curr_text, timepoint='prior')
        curr_prompt_embeds, curr_prompt_attns = self.prompt_wrap(img_embeds2, perception_embed, prev_text, timepoint='curr')

        # training targets
        prev_text = [t + self.end_sym for t in samples["prev_text"]]
        curr_text = [t + self.end_sym for t in samples["curr_text"]]
        prev_to_regress_tokens, prev_to_regress_embeds, prev_targets = self.training_input_generate(prev_text, prev_prompt_attns.shape[1], device)
        curr_to_regress_tokens, curr_to_regress_embeds, curr_targets = self.training_input_generate(curr_text, curr_prompt_attns.shape[1], device)

        # begin of sentence token
        bos = torch.ones([batch_size, 1],
                         dtype=prev_to_regress_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_attns = prev_prompt_attns[:, :1]

        # training inputs
        prev_inputs_embeds = torch.cat([bos_embeds, prev_prompt_embeds, prev_to_regress_embeds], dim=1)
        curr_inputs_embeds = torch.cat([bos_embeds, curr_prompt_embeds, curr_to_regress_embeds], dim=1)
        prev_attention_mask = torch.cat([bos_attns, prev_prompt_attns, prev_to_regress_tokens.attention_mask], dim=1)
        curr_attention_mask = torch.cat([bos_attns, curr_prompt_attns, curr_to_regress_tokens.attention_mask], dim=1)
        prev_outputs = self.llama_model(
            inputs_embeds=prev_inputs_embeds,
            attention_mask=prev_attention_mask,
            return_dict=True,
            labels=prev_targets,
        )
        curr_outputs = self.llama_model(
            inputs_embeds=curr_inputs_embeds,
            attention_mask=curr_attention_mask,
            return_dict=True,
            labels=curr_targets,
        )

        # progression alignment loss
        prog_loss = self.progression_loss(prog_text, perception_embed)

        # disease alignment loss
        prev_disease_loss = self.ce_align(img_embeds1, samples["prev_text"])
        curr_disease_loss = self.ce_align(img_embeds2, samples["curr_text"])

        loss = prev_outputs.loss + curr_outputs.loss + 0.1 * prog_loss + 0.5 * (prev_disease_loss + curr_disease_loss)
        return {"loss": loss, 'autoreg_loss': prev_outputs.loss + curr_outputs.loss, 'prog_loss': prog_loss, 'pathology_loss': prev_disease_loss + curr_disease_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res, val_score):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad}
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
                                                                          eval_res['Bleu_4'],eval_res['CIDEr']),
        )

        self.print("[Epoch {}] Saving checkpoint at step {} to {}.".format(self.trainer.current_epoch, global_step, save_to))
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

        image1 = samples["prev_image"]
        image2 = samples["curr_image"]
        prev_text = samples["prev_text"]
        curr_text = samples["curr_text"]

        # pathology encoding
        img_embeds1 = self.encode_img(image1)
        img_embeds1 = self.layer_norm(img_embeds1)
        img_embeds2 = self.encode_img(image2)
        img_embeds2 = self.layer_norm(img_embeds2)
        device = img_embeds1.device
        batch_size = img_embeds1.shape[0]

        # progression encoding
        p_frame = self.perception_frame.expand(batch_size, -1, -1, -1, -1)
        video = torch.stack([image1[0], p_frame[:, 0], p_frame[:, 1], image2[0]], dim=1)
        perception_embed = self.video_encoder(video)[:, 1:3, :, :]
        alpha = torch.sigmoid(self.perception_agg(perception_embed).squeeze(1))
        perception_embed = alpha * perception_embed[:, 0, :, :] + (1 - alpha) * perception_embed[:, 1, :, :]
        perception_embed = self.video_layer_norm(self.video_linear(perception_embed))

        # prompt wrapping
        prev_prompt_embeds, prev_prompt_attns = self.prompt_wrap(img_embeds1, perception_embed, curr_text, timepoint='prior')
        curr_prompt_embeds, curr_prompt_attns = self.prompt_wrap(img_embeds2, perception_embed, prev_text, timepoint='curr')

        # begin of sentence token
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_attns = prev_prompt_attns[:, :1]

        inputs_embeds = torch.cat([bos_embeds, curr_prompt_embeds], dim=1)
        attention_mask = torch.cat([bos_attns, curr_prompt_attns], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=None,
            top_p=None
        )

        # generating prior reports (not necessary)
        # prev_inputs_embeds = torch.cat([bos_embeds, prev_prompt_embeds], dim=1)
        # prev_attention_mask = torch.cat([bos_attns, prev_prompt_attns], dim=1)
        # prev_outputs = self.llama_model.generate(
        #     inputs_embeds=prev_inputs_embeds,
        #     attention_mask=prev_attention_mask,
        #     pad_token_id=self.llama_tokenizer.pad_token_id,
        #     num_beams=self.hparams.beam_size,
        #     do_sample=self.hparams.do_sample,
        #     min_new_tokens=self.hparams.min_new_tokens,
        #     max_new_tokens=self.hparams.max_new_tokens,
        #     repetition_penalty=self.hparams.repetition_penalty,
        #     length_penalty=self.hparams.length_penalty,
        #     temperature=None,
        #     top_p=None
        # )

        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

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
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref_1 = {k: [v] for k, v in zip(ids, ref)}
        hypo_1 = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref_1, hypo=hypo_1)
        eval_ce, ce_num = self.chexbert_metrics.compute(ref, hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        self.log_dict(eval_ce, sync_dist=True, logger=True)
        self.log_dict(ce_num, sync_dist=True, logger=True, reduce_fx=torch.sum)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        with open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w') as f1:
            json.dump(hypo_1, f1)
        with open(os.path.join(result_folder, 'refs.json'), 'w') as f2:
            json.dump(ref_1, f2)
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

        image1 = samples["prev_image"]
        image2 = samples["curr_image"]
        prev_text = samples["prev_text"]
        curr_text = samples["curr_text"]

        # pathology encoding
        img_embeds1 = self.encode_img(image1)
        img_embeds1 = self.layer_norm(img_embeds1)
        img_embeds2 = self.encode_img(image2)
        img_embeds2 = self.layer_norm(img_embeds2)
        device = img_embeds1.device
        batch_size = img_embeds1.shape[0]

        # progression encoding
        p_frame = self.perception_frame.expand(batch_size, -1, -1, -1, -1)
        video = torch.stack([image1[0], p_frame[:, 0], p_frame[:, 1], image2[0]], dim=1)
        perception_embed = self.video_encoder(video)[:, 1:3, :, :]
        alpha = torch.sigmoid(self.perception_agg(perception_embed).squeeze(1))
        perception_embed = alpha * perception_embed[:, 0, :, :] + (1 - alpha) * perception_embed[:, 1, :, :]
        perception_embed = self.video_layer_norm(self.video_linear(perception_embed))

        # prompt wrapping
        prev_prompt_embeds, prev_prompt_attns = self.prompt_wrap(img_embeds1, perception_embed, curr_text, timepoint='prior')
        curr_prompt_embeds, curr_prompt_attns = self.prompt_wrap(img_embeds2, perception_embed, prev_text, timepoint='curr')

        # begin of sentence token
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_attns = prev_prompt_attns[:, :1]

        inputs_embeds = torch.cat([bos_embeds, curr_prompt_embeds], dim=1)
        attention_mask = torch.cat([bos_attns, curr_prompt_attns], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=None,
            top_p=None
        )

        # Generating prior reports (not necessary)
        # prev_inputs_embeds = torch.cat([bos_embeds, prev_prompt_embeds], dim=1)
        # prev_attention_mask = torch.cat([bos_attns, prev_prompt_attns], dim=1)
        # prev_outputs = self.llama_model.generate(
        #     inputs_embeds=prev_inputs_embeds,
        #     attention_mask=prev_attention_mask,
        #     pad_token_id=self.llama_tokenizer.pad_token_id,
        #     num_beams=self.hparams.beam_size,
        #     do_sample=self.hparams.do_sample,
        #     min_new_tokens=self.hparams.min_new_tokens,
        #     max_new_tokens=self.hparams.max_new_tokens,
        #     repetition_penalty=self.hparams.repetition_penalty,
        #     length_penalty=self.hparams.length_penalty,
        #     temperature=None,
        #     top_p=None
        # )

        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref_1 = {k: [v] for k, v in zip(ids, ref)}
        hypo_1 = {k: [v] for k, v in zip(ids, hypo)}
        ref = [ref_1[id][0] for id in sorted(ref_1.keys())]
        hypo = [hypo_1[id][0] for id in sorted(hypo_1.keys())]

        eval_res = self.score(ref=ref_1, hypo=hypo_1)
        eval_ce, ce_num = self.chexbert_metrics.compute(ref, hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        self.log_dict(eval_ce, sync_dist=True, logger=True)
        self.log_dict(ce_num, sync_dist=True, logger=True, reduce_fx=torch.sum)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        with open(os.path.join(result_folder, f"test_result_localrank{self.local_rank}.json"), 'w') as f1:
            json.dump(hypo_1, f1)
        with open(os.path.join(result_folder, f'test_refs_localrank{self.local_rank}.json'), 'w') as f2:
            json.dump(ref_1, f2)

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