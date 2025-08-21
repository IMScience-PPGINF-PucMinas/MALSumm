import os
import json
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from .layers.summarizer import xLSTM
from utils.tensorboard_utils import TensorboardWriter

class Solver:
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains, and Evaluates the PGL-SUM model."""
        self.model = None
        self.optimizer = None
        self.writer = None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        self._set_random_seed()

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """Construct the PGL-SUM model and initialize its parameters."""
        self._initialize_model()
        self._initialize_optimizer_and_writer()

    def _initialize_model(self):
        """Initialize the xLSTM model."""
        self.model = xLSTM(
            input_size=self.config.input_size,
            output_size=self.config.input_size,
            num_segments=self.config.n_segments,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.config.device)

        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

    def _initialize_optimizer_and_writer(self):
        """Initialize the optimizer and Tensorboard writer."""
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req
            )
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """Initialize model weights."""
        for name, param in net.named_parameters():
            if 'weight' in name and param.dim() >= 2 and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented.")
            elif 'bias' in name or param.dim() < 2:
                nn.init.constant_(param, 0.1)

    def train(self):
        """Train the model."""
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()
            loss_history = self._train_one_epoch(epoch_i)
            self._log_epoch_results(epoch_i, loss_history)
            self.evaluate(epoch_i)

    def _train_one_epoch(self, epoch_i):
        """Train the model for one epoch."""
        loss_history = []
        num_batches = len(self.train_loader) // self.config.batch_size
        iterator = iter(self.train_loader)

        for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
            self.optimizer.zero_grad()
            batch_loss = self._process_batch(iterator)
            loss_history.append(batch_loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()

        return loss_history

    def _process_batch(self, iterator):
        """Process a single batch of data."""
        batch_loss = 0
        for _ in range(self.config.batch_size):
            frame_features, target = next(iterator)
            frame_features, target = frame_features.to(self.config.device), target.to(self.config.device)

            output, _ = self.model(frame_features.squeeze(0))
            output_adjusted = output.squeeze() if output.dim() > 1 else output.squeeze().mean(dim=1)

            loss = nn.MSELoss()(output_adjusted, target.squeeze(0))
            loss.backward()
            batch_loss += loss.item()

        return batch_loss / self.config.batch_size

    def _log_epoch_results(self, epoch_i, loss_history):
        """Log results for the current epoch."""
        mean_loss = np.mean(loss_history)
        print(f"Epoch {epoch_i} loss: {mean_loss:.4f}")

        if self.config.verbose:
            tqdm.write('Plotting...')

        self.writer.update_loss(mean_loss, epoch_i, 'loss_epoch')
        self._save_checkpoint(epoch_i)

    def _save_checkpoint(self, epoch_i):
        """Save model checkpoint."""
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch_i}.pkl')
        tqdm.write(f'Saving parameters at {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def evaluate(self, epoch_i, save_weights=False):
        """Evaluate the model."""
        self.model.eval()
        out_scores_dict = {}
        weights_save_path = os.path.join(self.config.score_dir, "weights.h5")

        for frame_features, video_name in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            scores, attn_weights = self._evaluate_video(frame_features)
            out_scores_dict[video_name] = scores

            if save_weights:
                self._save_attention_weights(weights_save_path, video_name, epoch_i, attn_weights)

        self._save_scores(out_scores_dict, epoch_i)

    def _evaluate_video(self, frame_features):
        """Evaluate a single video."""
        frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
        with torch.no_grad():
            scores, attn_weights = self.model(frame_features)
            scores = scores.squeeze(0).cpu().numpy().tolist()
            attn_weights = attn_weights.cpu().numpy()
        return scores, attn_weights

    def _save_attention_weights(self, weights_save_path, video_name, epoch_i, attn_weights):
        """Save attention weights."""
        with h5py.File(weights_save_path, 'a') as weights:
            weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)

    def _save_scores(self, out_scores_dict, epoch_i):
        """Save evaluation scores."""
        if not os.path.exists(self.config.score_dir):
            os.makedirs(self.config.score_dir)

        scores_save_path = os.path.join(self.config.score_dir, f"{self.config.video_type}_{epoch_i}.json")
        with open(scores_save_path, 'w') as f:
            if self.config.verbose:
                tqdm.write(f'Saving scores at {scores_save_path}')
            json.dump(out_scores_dict, f)
        os.chmod(scores_save_path, 0o777)


if __name__ == '__main__':
    pass