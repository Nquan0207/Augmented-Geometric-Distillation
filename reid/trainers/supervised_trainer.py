# -*- coding: utf-8 -*-
# Time    : 2021/8/6 11:13
# Author  : Yichen Lu


from .trainer import Trainer
from reid.utils.meters import AverageMeter, AverageMeters


class SupervisedTrainer(Trainer):
    def __init__(self,
                 networks,
                 optimizer,
                 lr_scheduler,
                 supcon_criterion=None,
                 supcon_weight=1.0,
                 triplet_weight=1.0,
                 **kwargs,
                 ):

        super(SupervisedTrainer, self).__init__()
        self.networks = networks
        self.trainables = [self.networks]
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.supcon_criterion = supcon_criterion
        self.supcon_weight = supcon_weight
        self.triplet_weight = triplet_weight

        self.meters = AverageMeters(AverageMeter("Batch Time"),
                                    AverageMeter("Pid Loss"),
                                    AverageMeter("Triplet Loss"),
                                    AverageMeter("SupCon Loss"),
                                    )

    def train(self, epoch, training_loader):
        self.before_train(epoch)

        for i, inputs in enumerate(training_loader):
            inputs, pids = self._parse_data(inputs)
            losses = self.train_step(inputs, pids)
            self.meters.update([self.timer(),
                                *losses,
                                ])
            print(f"Epoch: [{epoch}][{i + 1}/{len(training_loader)}], " + self.meters())

        self.after_train()

    def before_train(self, epoch):
        super(SupervisedTrainer, self).before_train()
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs, pids):
        self.optimizer.zero_grad()
        outputs = self.networks(inputs)
        loss, losses = self._compute_loss(outputs, pids)
        loss.backward()
        self.optimizer.step()
        return losses

    def _compute_loss(self, outputs, pids):
        pooled, preds = outputs["global"], outputs["preds"]
        pid_loss, triplet_loss = self.basic_criterion(pooled, preds, pids)
        # Compute SupCon loss if criterion is provided
        if self.supcon_criterion is not None:
            features = outputs.get("features", None)
            if features is None:
                # fallback: unsqueeze to [B,1,D]
                features = pooled.unsqueeze(1)
            # cur_range: use all labels as anchors by default
            cur_range = (pids.min().item(), pids.max().item() + 1)
            supcon_loss = self.supcon_criterion(features, pids, cur_range)
        else:
            supcon_loss = 0.0
        # Combine losses with weights
        loss = pid_loss + self.triplet_weight * triplet_loss + self.supcon_weight * supcon_loss
        return loss, [pid_loss.item(), triplet_loss.item(), float(supcon_loss)]


class SupervisedXTrainer(SupervisedTrainer):
    def __init__(self,
                 networks,
                 optimizer,
                 lr_scheduler,
                 **kwargs,
                 ):

        super(SupervisedXTrainer, self).__init__(networks,
                                                 optimizer,
                                                 lr_scheduler,
                                                 **kwargs
                                                 )

    def before_train(self, epoch):
        super(SupervisedTrainer, self).before_train()
        self.lr_scheduler.step(epoch)

    def train_step(self, inputs_list, pids):
        self.optimizer.zero_grad()
        for inputs in inputs_list:
            outputs = self.networks(inputs)
            loss, losses = self._compute_loss(outputs, pids)
            (loss / len(inputs_list)).backward()
        self.optimizer.step()
        return losses

    def _parse_data(self, inputs):
        imgs_list, _, pids, _ = inputs
        inputs_list = [imgs.to(self.device) for imgs in imgs_list]
        pids = pids.to(self.device)
        return inputs_list, pids