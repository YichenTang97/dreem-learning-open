import copy
import time
import json
import os
from .regularization import regularizers
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm

from ..models.modulo_net.net import ModuloNet
from ..trainers import optimizers, loss_functions
from ..utils import score_functions


class Trainer:
    def __init__(
            self,
            net: ModuloNet,
            metrics=["cohen_kappa", "f1", "accuracy"],
            epochs=30,
            metric_to_maximize="accuracy",
            patience=None,
            batch_size=32,
            save_folder=None,
            loss=None,
            regularization=None,
            swa=None,
            optimizer=None,
            num_workers=0,
            net_methods=None,
            train_postfix_fraction=0.1,
    ):
        if optimizer is None:
            optimizer = {'type': 'adam', 'args': {'lr': 1e-3}}
        if loss is None:
            loss = {'type': 'cross_entropy', 'args': {}}

        self.net_methods = net_methods if net_methods is not None else []
        print('METHODS')
        print(self.net_methods)
        self.net = net
        print('####################')
        print("Device: ", net.device)
        print('Using:', num_workers, ' workers')
        print('Trainable params', sum(p.numel() for p in net.parameters() if p.requires_grad))
        print('Total params', sum(p.numel() for p in net.parameters() if p.requires_grad))

        print('####################')

        self.loss_function = loss_functions[loss['type']](**loss['args'])
        self.optimizer_params = optimizer
        self.swa_params = swa
        self.regularization = []
        if regularization is not None:
            for regularizer in regularization:
                self.regularization += [
                    regularizers[regularizer['type']](self.net, **regularizer['args'])]

        self.reset_optimizer()

        self.metrics = {
            score: score_function for score, score_function in score_functions.items()
            if score in metrics + [metric_to_maximize]
        }

        self.iterations = 0
        self.epochs = epochs
        self.metric_to_maximize = metric_to_maximize
        self.patience = patience if patience else epochs
        self.loss_values = []
        self.save_folder = save_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_postfix_fraction = float(train_postfix_fraction)

    def reset_optimizer(self):
        self.base_optimizer = optimizers[self.optimizer_params['type']](self.net.parameters(),
                                                                        **self.optimizer_params[
                                                                            'args'])

        if self.swa_params is not None:
            try:
                from torchcontrib.optim import SWA
            except ImportError as e:
                raise ImportError(
                    "Trainer requested SWA (swa in trainer args) but `torchcontrib` is not "
                    "installed. This repo's bundled experiments do not use SWA; remove "
                    "`swa` from trainer args, or install legacy torchcontrib against an "
                    "old PyTorch (not supported on Python 3.12)."
                ) from e
            self.optimizer = SWA(self.base_optimizer, **self.swa_params)
            self.swa = True
            self.averaged_weights = False
        else:
            self.optimizer = self.base_optimizer
            self.swa = False

    def on_batch_start(self):
        pass

    def on_epoch_end(self):
        pass

    def validate(self, validation_dataset, return_metrics_per_records=False, verbose=False):
        self.net.eval()
        if self.swa:
            self.optimizer.swap_swa_sgd()
            self.averaged_weights = not self.averaged_weights

        metrics_epoch = {
            metric: []
            for metric in self.metrics.keys()
        }
        metrics_per_records = {}
        hypnograms = {}
        predictions = self.net.predict_on_dataset(validation_dataset, return_prob=False,
                                                  verbose=verbose)
        record_weights = []

        for record in validation_dataset.records:
            metrics_per_records[record] = {}
            hypnogram_target = validation_dataset.hypnogram[record]
            hypnogram_predicted = predictions[record]

            hypnograms[os.path.split(record)[-2]] = {
                'predicted': hypnogram_predicted.astype(int).tolist(),
                'target': hypnogram_target.astype(int).tolist()}
            record_weights += [np.sum(hypnogram_target >= 0)]
            for metric, metric_function in self.metrics.items():
                metric_value = metric_function(hypnogram_target, hypnogram_predicted)
                metrics_per_records[record][metric] = metric_value
                metrics_epoch[metric].append(metric_value)

        record_weights = np.array(record_weights)
        for metric in metrics_epoch.keys():
            metrics_epoch[metric] = np.array(metrics_epoch[metric])
            record_weights_tp = record_weights[~np.isnan(metrics_epoch[metric])]
            metrics_epoch[metric] = metrics_epoch[metric][~np.isnan(metrics_epoch[metric])]

            try:
                metrics_epoch[metric] = np.average(metrics_epoch[metric], weights=record_weights_tp)
            except ZeroDivisionError:
                metrics_epoch[metric] = np.nan

            if self.metric_to_maximize == metric:
                value = metrics_epoch[metric]

        if self.swa:
            if self.averaged_weights:
                self.optimizer.swap_swa_sgd()
                self.averaged_weights = not self.averaged_weights

        if return_metrics_per_records:

            return metrics_epoch, value, metrics_per_records, hypnograms
        else:
            return metrics_epoch, value

    def train_on_batch(self, data, mask=-1):
        # 1. train network
        # Set network in train mode
        self.net.train()

        # Retrieve inputs
        args, hypnogram = self.net.get_args(data)
        device = self.net.device
        hypnogram = hypnogram.to(device)
        mask = hypnogram != mask
        hypnogram = hypnogram[mask]

        # zero the network parameters gradien
        self.optimizer.zero_grad()

        # forward + backward
        output = self.net.forward(*args)[0]
        output = output[mask]
        loss_train = self.loss_function(output, hypnogram)
        if self.regularization is not None:
            for regularizer in self.regularization:
                regularizer.regularized_all_param(loss_train)

        loss_train.backward()
        if isinstance(hypnogram, tuple):
            hypnogram = hypnogram[0]

        self.iterations += 1
        return output, loss_train, hypnogram

    def validate_on_batch(self, data, mask=-1):
        # 2. Evaluate network on validation
        # Set network in eval mode
        self.net.eval()
        # Retrieve inputs
        args, hypnogram = self.net.get_args(data)
        device = self.net.device
        hypnogram = hypnogram.to(device)
        mask = hypnogram != mask
        hypnogram = hypnogram[mask]
        # forward
        output = self.net.forward(*args)[0]
        output = output[mask]
        loss_validation = self.loss_function(output, hypnogram)
        if isinstance(hypnogram, tuple):
            hypnogram = hypnogram[0]

        return output, loss_validation, hypnogram

    def train(self, train_dataset, validation_dataset, verbose=1, reset_optimizer=True):
        """
        Each epoch:
            - Run the full training DataLoader; optionally show tqdm with training loss and
              metrics. Postfix updates (loss + sklearn metrics on buffered predictions) occur
              every ``train_postfix_fraction`` of the epoch length in batches, and once at
              epoch end if a partial buffer remains.
            - Call ``validate(validation_dataset)`` once: full-record predictions, metrics
              aggregated per record then weighted-averaged (same as early stopping signal).
            - If ``metric_to_maximize`` improves vs the best so far, save ``best_net``;
              otherwise increment patience and stop when patience is exceeded.
        """
        if reset_optimizer:
            self.reset_optimizer()

        if self.save_folder:
            self.save_weights('best_net')

        loader_kw = dict(
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        if self.num_workers > 0:
            loader_kw['persistent_workers'] = True
        dataloader_train = DataLoader(train_dataset, **loader_kw)

        metrics_final = {
            metric: 0
            for metric in self.metrics.keys()
        }

        best_value = 0
        counter_patience = 0
        for epoch in range(0, self.epochs):
            if verbose == 0:
                print('EPOCH:', epoch)
            # Cumulative batch loss sum on device (avoid per-batch .item() sync).
            running_loss_sum = torch.zeros((), device=self.net.device, dtype=torch.float32)
            running_metrics = {metric: 0 for metric in self.metrics.keys()}
            # Per-batch GPU tensors; NumPy + sklearn only when flushing metrics.
            buffer_outputs_train = ([], [])
            if verbose > 0:
                # Configurate progress bar
                t = tqdm(dataloader_train, 0)
                t.set_description("EPOCH {}".format(epoch))
                update_postfix_every = max(
                    int(len(t) * self.train_postfix_fraction), 1)
                counter_update_postfix = 0
            else:
                t = dataloader_train

            def _flush_train_metrics_postfix(batch_count):
                """Accumulate sklearn metrics and refresh tqdm (non-empty buffers)."""
                nonlocal buffer_outputs_train, counter_update_postfix
                pred_np = torch.cat(buffer_outputs_train[0]).cpu().numpy()
                true_np = torch.cat(buffer_outputs_train[1]).cpu().numpy()
                for metric_name, metric_function in self.metrics.items():
                    running_metrics[metric_name] += metric_function(
                        pred_np, true_np
                    )
                buffer_outputs_train = ([], [])
                counter_update_postfix += 1
                if verbose > 0:
                    loss_sum_float = running_loss_sum.item()
                    t.set_postfix(
                        loss=loss_sum_float / float(batch_count),
                        **{
                            k: v / counter_update_postfix
                            for k, v in running_metrics.items()
                        },
                    )
                    self.loss_values.append((loss_sum_float, batch_count))

            t_start_train = time.time()
            n_batches_seen = 0
            for i, data in enumerate(t):
                self.on_batch_start()

                if verbose > 0:
                    if (i + 1) % update_postfix_every == 0 and i != 0:
                        if buffer_outputs_train[0]:
                            _flush_train_metrics_postfix(i + 1)

                # train
                output, loss_train, hypnogram = self.train_on_batch(data)

                running_loss_sum = running_loss_sum + loss_train.detach().float()
                n_batches_seen = i + 1
                if verbose > 0:
                    buffer_outputs_train[0].append(output.max(1)[1].detach())
                    buffer_outputs_train[1].append(hypnogram.detach())

                # gradient descent
                self.optimizer.step()
            if verbose > 0 and buffer_outputs_train[0]:
                _flush_train_metrics_postfix(n_batches_seen)
            t_stop_train = time.time()

            t_start_validation = time.time()
            metrics_epoch, value = self.validate(validation_dataset=validation_dataset)
            t_stop_validation = time.time()

            metrics_epoch["training_duration"] = t_stop_train - t_start_train
            metrics_epoch["validation_duration"] = t_stop_validation - t_start_validation

            if self.save_folder:
                self.save_weights(str(epoch) + "_net")

                json.dump(metrics_epoch,
                          open(os.path.join(self.save_folder, str(epoch) + "_metrics_epoch.json"), "w"))

            if value > best_value:
                print("New best {} !".format(self.metric_to_maximize), value)
                # best_net = copy.deepcopy(self.net)
                metrics_final = {
                    metric: metrics_epoch[metric]
                    for metric in self.metrics.keys()
                }
                best_value = value
                counter_patience = 0
                if self.save_folder:
                    self.save_weights('best_net')
                    json.dump(metrics_epoch,
                              open(os.path.join(self.save_folder, "metrics_best_epoch.json"), "w"))
            else:
                counter_patience += 1

            if counter_patience > self.patience:
                break

            self.on_epoch_end()
        return metrics_final

    def save_weights(self, file_name):
        self.net.save(os.path.join(self.save_folder, file_name))
