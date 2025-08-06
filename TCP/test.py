import torch
import torch.nn as nn
# from TCP.model_discrete import TCP
from TCP.model import TCP
from TCP.config import GlobalConfig
from collections import OrderedDict
from torch.utils.data import DataLoader
from TCP.data import CARLA_Data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from vlm_planner.planer_utils import Model_init, input_processing_real_batch, input_processing_carla_batch


config = GlobalConfig()
net = TCP(config)
ckpt = torch.load('/home/automan/ll/Bench2DriveZoo/ckpt/best_tcp.pth', map_location="cuda")
# ckpt = ckpt["state_dict"]
ckpt = ckpt['weight']
new_state_dict = OrderedDict()
for key, value in ckpt.items():
	new_key = key.replace("model.","")
	new_state_dict[new_key] = value
net.load_state_dict(new_state_dict, strict = False)
net.cuda()
net.eval()

config.val_data = 'tcp_bench2drive-val.npy'
val_set = CARLA_Data(root=config.root_dir_all, data_path=config.val_data, img_aug=False)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

llava_dir = "/media/automan/624d9dc2-1bdc-4d8d-a595-03ce40e57140/ll_projects/pretrained/LLaVA-Lightning-7B-delta-v1-1"
vision_tower = "/media/automan/624d9dc2-1bdc-4d8d-a595-03ce40e57140/ll_projects/pretrained/clip-vit-large-patch14"
torch_dtype = torch.bfloat16

clip_image_processor, tokenizer, LCB_model = Model_init(vision_tower, llava_dir,
														torch_dtype)
LCB_model.resize_token_embeddings(len(tokenizer))
device = "cuda" if torch.cuda.is_available() else "cpu"
LCB_model = LCB_model.to(device)

LCB_checkpoint = "/home/automan/ll/Bench2DriveZoo/ckpt/pytorch_model.bin"
state_dict = torch.load(LCB_checkpoint, map_location='cpu')
LCB_model.load_state_dict(state_dict)
del state_dict

# Iterate over the validation set
l2_05 = []
l2_1 = []
l2_15 = []
l2_2 = []
length = val_set.__len__()

command_map = {
	0: 'LEFT',
	1: 'RIGHT',
	2: 'STRAIGHT',
	3: 'LANE FOLLOW',
	4: 'CHANGE LANE LEFT',
	5: 'CHANGE LANE RIGHT'
}

with torch.no_grad():
	for index, batch in enumerate(tqdm(val_loader)):
		front_img = batch['front_img'].to('cuda')
		speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		state = torch.cat([speed, target_point, command], 1).to('cuda')
		gt_waypoints = batch['waypoints']

		command_text = [command_map[vec.argmax().item()] + " " + "<image>" for vec in command]
		image_clip, image, input_ids, attention_masks, targets = input_processing_carla_batch(
			image_tensor=front_img,
			command_list=command_text,
			clip_image_processor=clip_image_processor,
			tokenizer=tokenizer)
		_, pred_actions_embedding = LCB_model.evaluate(
			images_clip=image_clip,
			input_ids=input_ids,
			attention_masks=attention_masks,
		)
		pred = net.evaluate(front_img, state, target_point.to('cuda'),pred_actions_embedding)
		l2_05.extend(np.linalg.norm(pred['pred_wp'][:, 0].detach().cpu().numpy() - gt_waypoints[:, 0].numpy(), axis=1).tolist())
		l2_1.extend(np.linalg.norm(pred['pred_wp'][:, 1].detach().cpu().numpy() - gt_waypoints[:, 1].numpy(), axis=1).tolist())
		l2_15.extend(np.linalg.norm(pred['pred_wp'][:, 2].detach().cpu().numpy() - gt_waypoints[:, 2].numpy(), axis=1).tolist())
		l2_2.extend(np.linalg.norm(pred['pred_wp'][:, 3].detach().cpu().numpy() - gt_waypoints[:, 3].numpy(), axis=1).tolist())

print((sum(l2_05)/length + sum(l2_1)/length + sum(l2_15)/length + sum(l2_2)/length)/4)
