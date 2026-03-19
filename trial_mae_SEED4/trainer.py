import math, sys
import torch
import utils as ut
import numpy as np
import time

inf = float('inf')

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class MAETrainer:
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model
        self.device = device
        # Fix deprecation warning
        self._scaler = torch.amp.GradScaler('cuda')

    def train_epoch(self, train_loader, optimizer):
        for batch_idx, (eeg_windows, _) in enumerate(train_loader):
            # Fix deprecation warning
            with torch.amp.autocast('cuda', enabled=True):
                loss, pred, target, mask_corr = self.model(eeg_windows)


def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                        loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                        img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    for data_iter_step, (data_dcit) in enumerate(data_loader):
        
        # we use a per iteration (instead of per epoch) lr scheduler
        # print(data_iter_step)
        # print(len(data_loader))
        
        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)
        samples = data_dcit['eeg']
        
        img_features = None
        valid_idx = None
        if img_feature_extractor is not None:
            images = data_dcit['image']
            valid_idx = torch.nonzero(images.sum(dim=(1,2,3)) != 0).squeeze(1)
            img_feature_extractor.eval()
            with torch.no_grad():
                img_features = img_feature_extractor(preprocess(images[valid_idx]).to(device))['layer2']
        samples = samples.to(device)
        # img_features = img_features.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=True):
            loss, pred, _ = model(samples, img_features, valid_idx=valid_idx, mask_ratio=config.mask_ratio)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        # Fix: pred is ALREADY unpatchified by model.forward()
        # DO NOT call unpatchify again
        pred = pred.to('cpu').detach()
        samples = samples.to('cpu').detach()
        # pred is now [N, C, T] - same shape as samples
        
        # Compute correlation on full reconstruction
        pred_flat = pred.reshape(-1)
        samples_flat = samples.reshape(-1)
        
        # Compute Pearson correlation
        if torch.std(pred_flat) > 1e-6 and torch.std(samples_flat) > 1e-6:
            pred_norm = (pred_flat - pred_flat.mean()) / (pred_flat.std() + 1e-8)
            samples_norm = (samples_flat - samples_flat.mean()) / (samples_flat.std() + 1e-8)
            cor = (pred_norm * samples_norm).mean().item()
        else:
            cor = 0.0
        
        optimizer.zero_grad()

        total_loss.append(loss_value)
        total_cor.append(cor)
        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr, 'cor', np.mean(total_cor))

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time)/60.0, step=epoch)
    if config.local_rank == 0:        
        print(f'[Epoch {epoch}] loss: {np.mean(total_loss)}')

    return np.mean(total_loss), np.mean(total_cor)