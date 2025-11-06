# train_kd.py

import argparse
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import numpy as np
import time
from datetime import datetime
import torch.nn.functional as F
import random

# Import models
from Networks import FFNet
from Networks.FFNet_S import FFNet_S

# Import KD dataloader
from datasets_kd import Crowd_qnrf, Crowd_nwpu, Crowd_sh

# Import utils
from utils.pytorch_utils import Save_Handle, AverageMeter
import utils.log_utils as log_utils
try:
    import wandb
except ImportError:
    print("W&B not installed. Run 'pip install wandb' to log.")
    wandb = None

ARCH_NAMES = FFNet.__all__

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    gt_dmaps = torch.stack(transposed_batch[1], 0)
    return images, gt_dmaps

class KDTrainer(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        args = self.args
        sub_dir = f"KD_FFNet_S_{args.dataset}_{args.run_name}"
        self.save_dir = os.path.join("ckpts", sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        time_str = datetime.strftime(datetime.now(), "%m%d-%H%M%S")
        self.logger = log_utils.get_logger(
            os.path.join(self.save_dir, f"train_kd-{time_str}.log")
        )
        log_utils.print_config(vars(args), self.logger)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1, "KD training script only supports 1 GPU."
            self.logger.info("using 1 gpu")
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = 8

        # --- Load Datasets ---
        # NOTE: Using datasets_kd.py
        if args.dataset.lower() == "qnrf":
            dataset_class = Crowd_qnrf
        elif args.dataset.lower() == "nwpu":
            dataset_class = Crowd_nwpu
        elif args.dataset.lower() == "sha" or args.dataset.lower() == "shb":
            dataset_class = Crowd_sh
        else:
            raise NotImplementedError
            
        self.datasets = {
            'train': dataset_class(
                os.path.join(args.data_dir, 'train' if args.dataset.lower() == 'nwpu' else 'train_data'),
                args.crop_size, self.downsample_ratio, 'train'
            ),
            'val': dataset_class(
                os.path.join(args.data_dir, 'val' if args.dataset.lower() == 'nwpu' else 'test_data'),
                args.crop_size, self.downsample_ratio, 'val'
            )
        }
        
        # Handle QNRF's 'val' set being in the 'train' folder
        if args.dataset.lower() == 'qnrf':
            self.datasets['train'] = dataset_class(
                os.path.join(args.data_dir, 'train'),
                args.crop_size, self.downsample_ratio, 'train'
            )
            self.datasets['val'] = dataset_class(
                os.path.join(args.data_dir, 'val'),
                args.crop_size, self.downsample_ratio, 'val'
            )


        self.dataloaders = {
            x: DataLoader(
                self.datasets[x],
                collate_fn=(train_collate if x == "train" else default_collate),
                batch_size=(args.batch_size if x == "train" else 1),
                shuffle=(True if x == "train" else False),
                num_workers=args.num_workers,
                pin_memory=(True if x == "train" else False),
            )
            for x in ["train", "val"]
        }

        # --- Load Teacher Model (FFNet) ---
        self.teacher = FFNet.FFNet()
        self.teacher.to(self.device)
        if not args.teacher_weights:
            raise Exception("Must provide --teacher_weights path (e.g., SHA_model.pth)")
        self.logger.info(f"Loading teacher weights from {args.teacher_weights}")
        self.teacher.load_state_dict(torch.load(args.teacher_weights, self.device))
        self.teacher.eval() # Set teacher to eval mode
        for param in self.teacher.parameters():
            param.requires_grad = False # Freeze teacher

        # --- Load Student Model (FFNet-S) ---
        self.student = FFNet_S()
        self.student.to(self.device)
        self.optimizer = optim.AdamW(
            self.student.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.start_epoch = 0

        if args.resume:
            self.logger.info("loading pretrained student from " + args.resume)
            checkpoint = torch.load(args.resume, self.device)
            self.student.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint["epoch"] + 1

        # --- Define Losses (from paper) ---
        # L_hard: MSE between student output and GT density map
        self.l_hard = nn.MSELoss().to(self.device)
        # L_soft: L2 loss (MSE) between student and teacher density maps
        self.l_soft = nn.MSELoss().to(self.device)
        
        # L_feat is omitted for simplicity, as paper shows it has less impact than L_soft
        
        self.save_list = Save_Handle(max_num=1)
        self.best_mae = np.inf
        self.best_mse = np.inf
        
        # --- W&B Logging ---
        if args.wandb and wandb:
            self.wandb_run = wandb.init(
                config=args, project="FFNet-KD", name=args.run_name
            )
        else:
            if wandb:
                wandb.init(mode="disabled")

    def train(self):
        """training process"""
        args = self.args
        for epoch in range(self.start_epoch, args.max_epoch + 1):
            self.logger.info(
                "-" * 5 + "Epoch {}/{}".format(epoch, args.max_epoch) + "-" * 5
            )
            self.epoch = epoch
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()

    def train_epoch(self):
        epoch_loss = AverageMeter()
        epoch_l_hard = AverageMeter()
        epoch_l_soft = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.student.train() # Set student to train mode

        for step, (inputs, gt_dmaps) in enumerate(self.dataloaders["train"]):
            inputs = inputs.to(self.device)
            gt_dmaps = gt_dmaps.to(self.device)
            N = inputs.size(0)

            # Downsample GT density map to match model output
            gt_dmaps_down = F.avg_pool2d(gt_dmaps, self.downsample_ratio) * (self.downsample_ratio**2)
            gd_count = gt_dmaps.view(N, -1).sum(1)

            with torch.set_grad_enabled(True):
                # --- Forward passes ---
                # Teacher (frozen)
                with torch.no_grad():
                    D_T, _ = self.teacher(inputs)
                    D_T = D_T.detach() # Ensure no grads
                
                # Student (trainable)
                D_S, _ = self.student(inputs)
                
                # --- Calculate Losses (from paper) ---
                # L_hard (Student vs GT)
                loss_hard = self.l_hard(D_S, gt_dmaps_down)
                
                # L_soft (Student vs Teacher)
                loss_soft = self.l_soft(D_S, D_T)
                
                # --- Total Loss (with two-stage training) ---
                # Paper [cite: 183] "trained S using only L_hard for the first five epochs"
                if self.epoch < self.args.stage1_epochs:
                    L_total = self.args.a * loss_hard
                    loss_soft_val = torch.tensor(0.0)
                else:
                    # Paper [cite: 181] alpha=1, beta=0.5
                    L_total = (self.args.a * loss_hard) + (self.args.b * loss_soft)
                    loss_soft_val = loss_soft.item()

                self.optimizer.zero_grad()
                L_total.backward()
                self.optimizer.step()

                # --- Logging ---
                pred_count = torch.sum(D_S.view(N, -1), dim=1).detach().cpu().numpy()
                gt_count_cpu = gd_count.detach().cpu().numpy()
                pred_err = pred_count - gt_count_cpu
                
                epoch_loss.update(L_total.item(), N)
                epoch_l_hard.update(loss_hard.item(), N)
                epoch_l_soft.update(loss_soft_val, N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)

        # --- End of Epoch Logging ---
        if self.args.wandb and wandb:
            wandb.log({
                "train/TOTAL_loss": epoch_loss.get_avg(),
                "train/L_hard": epoch_l_hard.get_avg(),
                "train/L_soft": epoch_l_soft.get_avg(),
                "train/MAE": epoch_mae.get_avg(),
                "train/MSE": np.sqrt(epoch_mse.get_avg()),
            }, step=self.epoch)

        self.logger.info(
            f"Epoch {self.epoch} Train | Loss: {epoch_loss.get_avg():.2f} | "
            f"L_hard: {epoch_l_hard.get_avg():.2f} | L_soft: {epoch_l_soft.get_avg():.2f} | "
            f"MSE: {np.sqrt(epoch_mse.get_avg()):.2f} | MAE: {epoch_mae.get_avg():.2f} | "
            f"Cost {time.time() - epoch_start:.1f} sec"
        )
        
        model_state_dic = self.student.state_dict()
        save_path = os.path.join(self.save_dir, f"{self.epoch}_ckpt.tar")
        torch.save(
            {
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": model_state_dic,
            },
            save_path,
        )
        self.save_list.append(save_path) # Deletes old checkpoints

    def val_epoch(self):
        # Validation uses the same patch-based logic as original repo
        args = self.args
        epoch_start = time.time()
        self.student.eval()  # Set student to evaluate mode
        epoch_res = []

        for inputs, count, name in self.dataloaders["val"]:
            with torch.no_grad():
                inputs = inputs.to(self.device)
                crop_imgs, crop_masks = [], []
                b, c, h, w = inputs.size()
                rh, rw = args.crop_size, args.crop_size
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                        mask = torch.zeros([b, 1, h, w]).to(self.device)
                        mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                        crop_masks.append(mask)
                crop_imgs, crop_masks = map(
                    lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
                )

                crop_preds = []
                nz, bz = crop_imgs.size(0), args.batch_size
                for i in range(0, nz, bz):
                    gs, gt = i, min(nz, i + bz)
                    crop_pred, _ = self.student(crop_imgs[gs:gt]) # Use student model

                    _, _, h1, w1 = crop_pred.size()
                    crop_pred = (
                        F.interpolate(
                            crop_pred,
                            size=(h1 * 8, w1 * 8),
                            mode="bilinear",
                            align_corners=True,
                        ) / 64.0
                    )
                    crop_preds.append(crop_pred)
                crop_preds = torch.cat(crop_preds, dim=0)

                idx = 0
                pred_map = torch.zeros([b, 1, h, w]).to(self.device)
                for i in range(0, h, rh):
                    gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
                    for j in range(0, w, rw):
                        gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                        pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                        idx += 1
                
                mask = crop_masks.sum(dim=0).unsqueeze(0)
                outputs = pred_map / mask

                res = count[0].item() - torch.sum(outputs).item()
                epoch_res.append(res)
                
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))

        self.logger.info(
            f"Epoch {self.epoch} Val | MSE: {mse:.2f} | MAE: {mae:.2f} | "
            f"Cost {time.time() - epoch_start:.1f} sec"
        )

        if self.args.wandb and wandb:
            wandb.log({"val/MSE": mse, "val/MAE": mae}, step=self.epoch)

        model_state_dic = self.student.state_dict()
        if mae < self.best_mae:
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info(
                f"SAVE BEST MODEL: MAE {self.best_mae:.2f} MSE {self.best_mse:.2f} at epoch {self.epoch}"
            )
            model_path = os.path.join(self.save_dir, "best_model_mae.pth")
            torch.save(model_state_dic, model_path)
            
            if self.args.wandb and wandb and self.wandb_run:
                artifact = wandb.Artifact("FFNet-S-model", type="model")
                artifact.add_file(model_path)
                self.wandb_run.log_artifact(artifact)

def parse_args():
    parser = argparse.ArgumentParser(description='Train FFNet-S with KD')
    parser.add_argument('--data-dir', default='/datasets/shanghaitech/part_A_final', help='data path')
    parser.add_argument('--dataset', default='sha', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--teacher-weights', default='', type=str, required=True,
                        help='the path to the pretrained teacher model (e.g., SHA_model.pth)')
    
    # KD parameters
    parser.add_argument('--a', type=float, default=1.0, help='Weight for L_hard (alpha in paper)')
    parser.add_argument('--b', type=float, default=0.5, help='Weight for L_soft (beta in paper)')
    parser.add_argument('--stage1-epochs', type=int, default=5, help='Num epochs to train on L_hard only')
    
    # Standard training parameters
    parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0, help='the weight decay')
    parser.add_argument('--resume', default='', type=str, help='the path of resume training student model')
    parser.add_argument('--max-epoch', type=int, default=1000, help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5, help='run validation every n epochs')
    parser.add_argument('--val-start', type=int, default=100, help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=4, help='train batch size (keep small for 6GB VRAM!)')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=4, help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=256, help='the crop size of the train image')
    
    # Logging
    parser.add_argument('--run-name', default='ffnet-s-kd-run1', help='run name for wandb interface/logging')
    parser.add_argument('--wandb', default=1, type=int, help='boolean to set wandb logging')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    # Set crop size based on dataset
    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
    elif args.dataset.lower() == 'sha':
        args.crop_size = 256
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    
    return args

if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()
    
    trainer = KDTrainer(args)
    trainer.setup()
    trainer.train()