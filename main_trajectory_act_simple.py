"""Main script for trajectory optimization."""

import os
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.distributions import Beta


from TCP.config import GlobalConfig

# from datasets.dataset_engine import RLBenchDataset
from engine_act_simple import BaseTrainTester
# from diffuser_actor import DiffuserActorACTS

from vlm_planner.utils.common_utils import (
    count_parameters
)
from pathlib import Path
#
# from datasets.calvin_dataset import transfer
# from planer_utils import input_processing_real_batch
from vlm_planner.planer_utils import input_processing_real_batch, input_processing_carla_batch

# class Arguments(tap.Tap):
#     id: str = "TCP"  #: Unique experiment identifier.
#     epochs: int = 60  #: Number of train epochs.
#     lr: float = 1e-4  #: Learning rate.
#     val_every: int = 3  #: Validation frequency (epochs).
#     batch_size: int = 5  #: Batch size.
#     logdir: Path = Path("log")  #: Directory to log data to.
#     gpus: int = 1  #: Number of GPUs.
#
#     def post_init(self):
#         """Called after parsing: automatically append `id` to `logdir`."""
#         # if you prefer a string:
#         # self.logdir = os.path.join(str(self.logdir), self.id)
#         # but since we annotated as Path we can do:
#         self.logdir = self.logdir / self.id

class TrainTester(BaseTrainTester):
    """Train/test a trajectory optimization algorithm."""

    def __init__(self, args):
        """Initialize."""
        super().__init__(args)

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        # instruction = load_instructions(
        #     self.args.instructions,
        #     tasks=self.args.tasks,
        #     variations=self.args.variations
        # )
        # if instruction is None:
        #     raise NotImplementedError()
        # else:
        #     taskvar = [
        #         (task, var)
        #         for task, var_instr in instruction.items()
        #         for var in var_instr.keys()
        #     ]

        # Initialize datasets with arguments
        self.config = GlobalConfig()

        if self.args.vlm_output_type == "traj":
            from TCP.data import CARLA_Data
        elif self.args.vlm_output_type == "occ":
            from TCP.data_occ import CARLA_Data

        # Initialize datasets with arguments
        train_dataset = CARLA_Data(root=self.config.root_dir_all,
                                   data_path=self.config.train_data,
                                   img_aug=self.config.img_aug)

        test_dataset = CARLA_Data(root=self.config.root_dir_all,
                                  data_path=self.config.val_data, )
        return train_dataset, test_dataset

    def get_model(self):
        """Initialize the model."""
        # Initialize model with arguments
        # from TCP.model import TCP
        from TCP.model_discrete import TCP
        _model = TCP(self.config)
        print("Model parameters:", count_parameters(_model))

        return _model

    @staticmethod
    def get_criterion():
        return TrajectoryCriterion()

    def train_one_step(self, model, criterion, optimizer, step_id,
                       sample, embedding, lmcliploss=None, act_pred=None):
        """Run a single training step."""
        aux_loss = 0
        speed = sample['speed'].to(dtype=torch.float32).view(-1, 1) / 12.
        target_point = sample['target_point'].to(dtype=torch.float32)
        command = sample['target_command']
        state = torch.cat([speed, target_point, command], 1)

        pred = model(
            img = sample["front_img"],
            state=state,
            target_point = target_point,
            embedding=embedding.cuda(),
        )

        gt_waypoints = sample['waypoints']
        action_loss = F.nll_loss(F.log_softmax(pred['action_index'], dim=1), sample['action_index'])
        speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()

        future_action_loss = 0
        for i in range(self.config.pred_len):
            action_loss = F.nll_loss(F.log_softmax(pred['future_action_index'][i], dim=1), sample['future_action_index'][i])
            future_action_loss += action_loss
        future_action_loss /= self.config.pred_len

        feature_in = False
        if feature_in:
            value = sample['value'].view(-1,1)
            feature = sample['feature']
            value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
            feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) +F.mse_loss(pred['pred_features_ctrl'], feature))* self.config.features_weight

            future_feature_loss = 0
            for i in range(self.config.pred_len):
                future_feature_loss += F.mse_loss(pred['future_feature'][i], sample['future_feature'][i]) * self.config.features_weight
                future_feature_loss /= self.config.pred_len

            loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss
        else:
            loss = action_loss+speed_loss+wp_loss+future_action_loss

        if act_pred is None:
            return loss
        else:
            act_pred, start_idx = act_pred
            if self.args.vlm_output_type =='traj':
                wp_llm_loss = F.l1_loss(act_pred, gt_waypoints, reduction='none').mean()
            elif self.args.vlm_output_type == 'occ':
                occ_target = sample['occ']
                wp_llm_loss = F.binary_cross_entropy_with_logits(act_pred, occ_target).mean()

            loss = 0.2*wp_llm_loss + loss

        if (step_id + 1) < self.args.train_iters:
            model.requires_grad_(False)
        else:
            model.requires_grad_(True)

        # action_loss = criterion.compute_loss(out)
        if lmcliploss is not None:
            loss = loss + lmcliploss
        else:
            loss = loss

        loss.backward()

        # Update
        if ((step_id + 1) > self.args.train_iters) and ((step_id + 1) % self.args.accumulate_grad_batches == 0):
            optimizer.step()
            optimizer.zero_grad()

        # Log
        if dist.get_rank() == 0 and (step_id + 1) % (0.01 * self.args.val_freq) == 0:
            self.writer.add_scalar("lr", self.args.lr, step_id)
            self.writer.add_scalar("action_loss", action_loss, step_id)
            self.writer.add_scalar("train-loss/noise_mse", loss, step_id)


    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, LCB_model, clip_image_processor,tokenizer,
                          step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break
            commands = sample['target_command']

            command_map = {
                0: 'LEFT',
                1: 'RIGHT',
                2: 'STRAIGHT',
                3: 'LANE FOLLOW',
                4: 'CHANGE LANE LEFT',
                5: 'CHANGE LANE RIGHT'
            }

            commands = [command_map[vec.argmax().item()] + " " + "<image>" for vec in commands]
            # fisrt stage================LLM===================
            start_idx = []
            initial_id = 0
            for i in range(len(commands)):
                if sample['waypoints'][i][0][0] == sample['waypoints'][i][3][0]:
                    initial_id = i
                #     if sample['curr_gripper_history'][i][0][0]==sample['curr_gripper_history'][i][2][0]:
                #         initial_id = i
                if (i - initial_id) % 1 == 0:
                    start_idx.append(i)
            # image_rgb = sample['rgbs'][:, 0]
            # image_select_rows = image_rgb[start_idx]
            # conv_select = [conversations[i] for i in start_idx]

            image_select_rows = sample['front_img']
            image_clip, image, input_ids, attention_masks, targets = input_processing_carla_batch(
                image_tensor=image_select_rows,
                command_list=commands,
                clip_image_processor=clip_image_processor,
                tokenizer=tokenizer)

            _, pred_actions_embedding = LCB_model.module.evaluate(
                images_clip=image_clip,
                input_ids=input_ids,
                attention_masks=attention_masks,
            )

            total_action_embedding = torch.zeros(len(commands), pred_actions_embedding.shape[1])  # batch, 512
            # print(total_action_embedding.shape)
            # total_action_embedding = torch.zeros_like(pred_actions_embedding,device=pred_actions_embedding.device)
            for i in range(len(commands)):
                if i in start_idx:
                    total_action_embedding[i] = pred_actions_embedding[start_idx.index(i)]
                else:
                    previous_idx = max([idx for idx in start_idx if idx < i])
                    total_action_embedding[i] = total_action_embedding[previous_idx]

            # revise 0305
            total_action_embedding = total_action_embedding.unsqueeze(1)

            speed = sample['speed'].to(dtype=torch.float32).view(-1, 1) / 12.
            target_point = sample['target_point'].to(dtype=torch.float32)
            command = sample['target_command']
            state = torch.cat([speed, target_point, command], 1)

            pred = model(
                gt=sample,
                img=sample["front_img"],
                state=state,
                target_point=target_point,
                embedding=total_action_embedding.to(device),
                vlm_output_type = "traj",
                run_inference=True
            )

            losses, losses_B = criterion.compute_metrics(
                pred,
                self.to_cuda(sample, device),
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], torch.Tensor([l]).to(device)])

            # Gather per-task statistics

            # Generate visualizations
            # if i == 0 and dist.get_rank() == 0 and step_id > -1:
            #     viz_key = f'{split}-viz/viz'
            #     viz = generate_visualizations(
            #         action,
            #         sample["trajectory"].to(device),
            #         sample["trajectory_mask"].to(device)
            #     )
            #     self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)

    @torch.no_grad()
    def evaluate_nsteps_lcb(self, model, criterion, loader, step_id, LCB_model, clip_image_processor, tokenizer, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        if self.args.val_iters != -1:
            val_iters = self.args.val_iters
        values = {}
        device = next(model.parameters()).device
        model.eval()

        for i, sample in enumerate(loader):
            if i == val_iters:
                break
            if self.args.keypose_only:
                sample["trajectory"] = sample["trajectory"][:, [-1]]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, [-1]]
            else:
                sample["trajectory"] = sample["trajectory"][:, 1:]
                sample["trajectory_mask"] = sample["trajectory_mask"][:, 1:]
            conversations, questions = transfer(sample['instr_text'])
            start_idx = []
            initial_id=0
            for x in range(len(conversations)):
                if sample['curr_gripper_history'][x][0][0]==sample['curr_gripper_history'][x][2][0]:
                    initial_id = x
                if (x-initial_id) % self.args.sample_rate == 0:
                    start_idx.append(x)
            image_rgb = sample['rgbs'][:, 0]
            image_select_rows = image_rgb[start_idx]
            conv_select = [questions[y] for y in start_idx]
            print("conv_select", conv_select)
            image_clip, image, input_ids, attention_masks, targets = input_processing_real_batch(image_tensor=image_select_rows, conv_list=conv_select, clip_image_processor=clip_image_processor, tokenizer=tokenizer)
            LCB_model.eval()

            output_ids, pred_actions_embedding = LCB_model.module.evaluate(image_clip, input_ids, max_new_tokens=512, tokenizer=tokenizer)
            print(pred_embeddings.shape)

            for z in range(len(conversations)):
                if z == 0:
                    total_action_embedding[z] = pred_actions_embedding[start_idx.index(0)]
                else:
                    previous_idx = 0
                    total_action_embedding[z] = total_action_embedding[previous_idx]
            total_action_embedding = total_action_embedding.unsqueeze(1)

############second stage
            curr_gripper = (
                sample["curr_gripper"] if self.args.num_history < 1
                else sample["curr_gripper_history"][:, -self.args.num_history:]
            )
            action = model(
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device),
                sample["rgbs"].to(device),
                sample["pcds"].to(device),
                total_action_embedding.to(device),
                # sample["instr"].to(device),
                curr_gripper.to(device),
                run_inference=True
            )
            losses, losses_B = criterion.compute_metrics(
                action,
                sample["trajectory"].to(device),
                sample["trajectory_mask"].to(device)
            )

            # Gather global statistics
            for n, l in losses.items():
                key = f"{split}-losses/mean/{n}"
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            # Gather per-task statistics
            tasks = np.array(sample["task"])
            for n, l in losses_B.items():
                for task in np.unique(tasks):
                    key = f"{split}-loss/{task}/{n}"
                    l_task = l[tasks == task].mean()
                    if key not in values:
                        values[key] = torch.Tensor([]).to(device)
                    values[key] = torch.cat([values[key], l_task.unsqueeze(0)])

            # Generate visualizations
            if i == 0 and dist.get_rank() == 0 and step_id > -1:
                viz_key = f'{split}-viz/viz'
                viz = generate_visualizations(
                    action,
                    sample["trajectory"].to(device),
                    sample["trajectory_mask"].to(device)
                )
                self.writer.add_image(viz_key, viz, step_id)

        # Log all statistics
        values = self.synchronize_between_processes(values)
        values = {k: v.mean().item() for k, v in values.items()}
        if dist.get_rank() == 0:
            if step_id > -1:
                for key, val in values.items():
                    self.writer.add_scalar(key, val, step_id)

            # Also log to terminal
            print(f"Step {step_id}:")
            for key, value in values.items():
                print(f"{key}: {value:.03f}")

        return values.get('val-losses/traj_pos_acc_001', None)
#
#
#
# def traj_collate_fn(batch):
#     keys = [
#         "trajectory", "trajectory_mask",
#         "rgbs", "pcds",
#         "curr_gripper", "curr_gripper_history", "action", "instr"
#     ]
#     ret_dict = {
#         key: torch.cat([
#             item[key].float() if key != 'trajectory_mask' else item[key]
#             for item in batch
#         ]) for key in keys
#     }
#
#     ret_dict["instr_text"] = []
#     ret_dict["task"] = []
#     for item in batch:
#         ret_dict["instr_text"] += item['instr_text']
#         ret_dict["task"] += item['task']
#     return ret_dict
#
#
class TrajectoryCriterion:

    def __init__(self):
        pass

    def compute_loss(self, pred, gt=None, is_loss=True):
        if not is_loss:
            assert gt is not None
            return self.compute_metrics(pred, gt)
        return pred

    @staticmethod
    def compute_metrics(pred, gt, pred_len=4, speed_weight=0.05,features_weight=0.05,value_weight = 0.001 ):
        # pred/gt are (B, L, 2), mask (B, L)
        speed = gt['speed'].to(dtype=torch.float32).view(-1, 1) / 12.
        dist_sup = Beta(gt['action_mu'], gt['action_sigma'])
        dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
        kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
        action_loss = torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5
        speed_loss = F.l1_loss(pred['pred_speed'], speed) * speed_weight

        # value = gt['value'].view(-1, 1)
        # value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'],
        #                                                                       value)) * value_weight
        # feature = gt['feature']
        # feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'],
        #
        #                                                                              feature)) * features_weight
        gt_waypoints = gt['waypoints']
        # future_feature_loss = 0
        # future_action_loss = 0
        # for i in range(pred_len):
        #     dist_sup = Beta(gt['future_action_mu'][i], gt['future_action_sigma'][i])
        #     dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
        #     kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
        #     future_action_loss += torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5
        #     future_feature_loss += F.mse_loss(pred['future_feature'][i],
        #                                       gt['future_feature'][i]) * features_weight
        # future_feature_loss /= pred_len
        # future_action_loss /= pred_len
        wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
        print('Actual waypoint：',gt_waypoints.cpu())
        print('Predicted waypoint：', pred['pred_wp'].cpu())

        # Trajectory metrics
        loss = {
            'action_loss': action_loss.item(),
             'wp_loss': wp_loss.item(),
            'speed_loss': speed_loss.item(),
            # 'future_action_loss': future_action_loss.item(),

        }
        # loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss+ future_feature_loss + future_action_loss
        print("action_loss:", loss['action_loss'])
        print("wp_loss:", loss['wp_loss'])
        print("speed_loss:", loss['speed_loss'])

        return loss, loss['action_loss']
#
#
# def fig_to_numpy(fig, dpi=60):
#     buf = io.BytesIO()
#     fig.savefig(buf, format="png", dpi=dpi)
#     buf.seek(0)
#     img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
#     buf.close()
#     img = cv2.imdecode(img_arr, 1)
#     return img
#
#
# def generate_visualizations(pred, gt, mask, box_size=0.3):
#     batch_idx = 0
#     pred = pred[batch_idx].detach().cpu().numpy()
#     gt = gt[batch_idx].detach().cpu().numpy()
#     mask = mask[batch_idx].detach().cpu().numpy()
#
#     fig = plt.figure(figsize=(10, 10))
#     ax = plt.axes(projection='3d')
#     ax.scatter3D(
#         pred[~mask][:, 0], pred[~mask][:, 1], pred[~mask][:, 2],
#         color='red', label='pred'
#     )
#     ax.scatter3D(
#         gt[~mask][:, 0], gt[~mask][:, 1], gt[~mask][:, 2],
#         color='blue', label='gt'
#     )
#
#     center = gt[~mask].mean(0)
#     ax.set_xlim(center[0] - box_size, center[0] + box_size)
#     ax.set_ylim(center[1] - box_size, center[1] + box_size)
#     ax.set_zlim(center[2] - box_size, center[2] + box_size)
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_zticklabels([])
#     plt.legend()
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#
#     img = fig_to_numpy(fig, dpi=120)
#     plt.close()
#     return img.transpose(2, 0, 1)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
    # parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate.')
    parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    # parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
    parser.add_argument('--gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--llava_dir', type=str,
                        default='/home/automan/ll/pretrained/LLaVA-Lightning-7B-delta-v1-1', help='llava')
    # parser.add_argument('--token_dir', type=str,
    #                     default='/home/users/ntu/yongxias/scratch/lilin_projects/pretrained/LLaVA-Lightning-7B-delta-v1-1', help='token')
    parser.add_argument('--vision_tower', type=str,
                        default='/home/automan/ll/pretrained/clip-vit-large-patch14', help='vision tower')
    parser.add_argument('--sample_rate', type=int, default=1, help='sample rate')
    parser.add_argument('--stage2_train_iters', type=int, default=10_000, help='stage2_train_iters')
    parser.add_argument('--pred_len', type=int, default=4, help='Length of predicted trajectory.')
    parser.add_argument('--train_iters', type=int, default=10_000, help='iteration number of first training.')
    parser.add_argument('--base_log_dir', type=str,
                        default=Path(__file__).parent / "train_logs", help='save log')
    parser.add_argument('--exp_log_dir', type=str,
                        default="exp", help='save log')
    parser.add_argument('--run_log_dir', type=str,
                        default="run", help='save log')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help=' ')
    parser.add_argument('--val_freq', type=int, default=1_000, help=' ')
    parser.add_argument('--eval_only', type=int, default=0, help=' ')
    parser.add_argument('--val_iters', type=int, default=5, help='iteration number of first training.')
    parser.add_argument('--vlm_output_type', type=str, default='occ', help='[traj, occ],the type of output from VLM')
    parser.add_argument('--training_checkpoint', type=str,
                        default='/home/automan/ll/Bench2DriveZoo/ckpt/tcp_b2d.ckpt', help='pretrained checkpoint')
    parser.add_argument('--LCB_checkpoint', type=str,
                        default='/home/automan/ll/Bench2DriveZoo/ckpt/pytorch_model.bin', help='pretrained checkpoint')




    args = parser.parse_args()
    log_dir = args.base_log_dir / args.exp_log_dir / args.run_log_dir
    args.log_dir = log_dir
    # args.logdir = os.path.join(args.logdir, args.id)
    print("Arguments:")
    print(args)
    print("-" * 100)

    print("Logging:", args.log_dir)
    print(
        "Available devices (CUDA_VISIBLE_DEVICES):",
        os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    print("Device count", torch.cuda.device_count())
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # Run
    train_tester = TrainTester(args)
    train_tester.main()
    # train_tester.main(collate_fn=traj_collate_fn)
