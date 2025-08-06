from typing import List
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel


from peft import LoraConfig, get_peft_model
from vlm_planner.model.llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM, LlavaLlamaModel)
from vlm_planner.datasets.utils_llcb import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

class LisaMetaModel:
    def __init__(self, config, **kwargs):
        super(LisaMetaModel, self).__init__(config)

        self.text_hidden_fcs = None
        self.config = config
        self.config.out_dim = kwargs["out_dim"]
        # self.config.out_dim = 512
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        self.initialize_lisa_modules(self.config)

    def initialize_lisa_modules(self, config):
        # Projection layer
        in_dim = config.hidden_size
        out_dim = self.config.out_dim
        text_fc = [nn.Linear(in_dim, out_dim)]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()

        # Action prediction
        self.pred_act_mlps = nn.Linear(in_dim, in_dim//2)
        self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=in_dim//2)
        self.pred_traj = nn.Linear(in_dim//2, 2)
        # self.pred_pos_act = nn.Linear(in_dim//2, 3) # arm action
        # self.pred_rot_act = nn.Linear(in_dim//2, 6) # arm action
        # self.pred_gripper_act = nn.Linear(in_dim//2, 1) # gripper action (binary)

        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True

        self.global_1d_pool = nn.AdaptiveAvgPool1d(1)


class LisaModel(LisaMetaModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(LisaModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class LISAForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
        # config.mm_vision_tower = kwargs.get("vision_tower", '/media/lin/New/pretrained/clip-vit-large-patch14')
        config.mm_vision_tower = kwargs.get("vision_tower", "openai/clip-vit-large-patch14")
        self.seg_token_idx = 32003

        super().__init__(config)

        self.model = LisaModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(self,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        target_point: torch.FloatTensor,
        attention_masks: torch.LongTensor,
        tokenizer,
        pred_len,
        **kwargs,
    ):
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx

        if torch.cuda.is_available():
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().cuda(), seg_token_mask],
                                       dim=1, )
        else:
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().to(torch.device("mps")), seg_token_mask],
                                       dim=1, )
        
         #[bs, 255+sequence_length] 255+82=337
        
        output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            output_hidden_states=True,
        )
        
        output_hidden_states = output.hidden_states 

        hidden_states = []

        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1].float()))
        action_latents = self.model.pred_act_mlps(output_hidden_states[-1][seg_token_mask].float())
        output_wp = list()
        z = action_latents
        x = torch.zeros(size=(z.shape[0], 2),
                        dtype=z.dtype).type_as(z)
        for _ in range(pred_len):
            x_in = torch.cat([x,
                              target_point.to(dtype=torch.float32).cuda()], dim=1)
            z = self.model.decoder_traj(x_in, z)
            dx = self.model.pred_traj(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)
        # pos_pred = self.model.pred_pos_act(action_latents)
        # rot_pred = self.model.pred_rot_act(action_latents)
        # gripper_pred = self.model.pred_gripper_act(action_latents)
        # act_pred = torch.cat([pos_pred,rot_pred,gripper_pred],dim=-1)

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[seg_token_mask]
        ce_loss = 0
        return pred_embeddings, ce_loss, pred_wp
        # return pred_embeddings, ce_loss, act_pred
    
    def evaluate(
        self,
        images_clip,
        input_ids,
        attention_masks,
        tokenizer=None
    ):
        with torch.no_grad():
            seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
            seg_token_mask = torch.cat([torch.zeros((seg_token_mask.shape[0], 256)).bool().cuda(), seg_token_mask], dim=1,) #[bs, 255+sequence_length] 255+82=337
            
            output = super().forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            output_hidden_states=True)
            output_hidden_states = output.hidden_states
            hidden_states = []
            
            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states.float()))
            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            pred_embeddings = last_hidden_state[seg_token_mask]
        
        return None, pred_embeddings


