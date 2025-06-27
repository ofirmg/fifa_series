import os
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
import plotly.express as px

from stat_predict.dataset.sp_dataset import StatsDataset
from stat_predict.dataset.utils import FeatureFamilies
from stat_predict.models.model import ModelArgs, MLStatPredict
from stat_predict.static.config import DefaultParameters, YOY_CATS


def compute_balanced_class_weights(y: pd.Series) -> torch.Tensor:
    """
    Compute balanced class weights for use in PyTorch models.

    Parameters:
    - df (pd.DataFrame): DataFrame containing labels.
    - label_col (str): Name of the column with binary labels (0 or 1).

    Returns:
    - torch.Tensor: A tensor of class weights [weight_for_class_0, weight_for_class_1].
    """
    label_counts = Counter(y)
    total_samples = len(y)
    num_classes = 2

    weights = []
    for i in range(num_classes):
        class_count = label_counts.get(i, 0)
        weight = total_samples / (num_classes * class_count) if class_count > 0 else 0.0
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)


class MLPNet(nn.Module):
    def __init__(self,
                 in_features: int,
                 num_years_back: int = 3,
                 h1: int = 128, h2: int = 64, h3: int = 32,
                 dropout=0.1,
                 activation=nn.ReLU(),
                 **kwargs
                 ):
        super(MLPNet, self).__init__()
        feature_per_year = in_features // num_years_back

        self.input_mode = '2d'
        self.n_y = num_years_back
        self.feature_per_year = feature_per_year
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3

        self.input_norm = nn.BatchNorm1d(num_years_back * feature_per_year)
        self.fc_1 = nn.Linear(feature_per_year * num_years_back, h1)
        self.fc_2 = nn.Linear(h1, h2)
        self.fc_3 = nn.Linear(h2, h3)

        self.drop = nn.Dropout(p=dropout)
        self.activation = activation
        self.final_fc = nn.Linear(h3, len(YOY_CATS))

    def forward(self, x):
        x = x.float()
        x = x.flatten(1)
        x = self.input_norm(x)
        x0 = self.drop(x)

        x0 = self.activation(self.fc_1(x0))
        x = self.activation(self.fc_2(x0))
        x = self.activation(self.fc_3(x))
        x = self.final_fc(x)
        return x


class RowEmbedder(nn.Module):
    def __init__(self, in_features: int, layers: list[int]):
        super().__init__()
        net = []
        for out_dim in layers:
            net.append(nn.Linear(in_features, out_dim))
            net.append(nn.ReLU())
            in_features = out_dim
        self.in_features = in_features
        self.model = nn.Sequential(*net)
        self.out_dim = layers[-1]

    def forward(self, x):
        # x shape: (batch, time, features)
        batch, time, features = x.shape
        x = x.view(-1, features)  # flatten batch and time
        x = self.model(x)
        x = x.view(batch, time, -1)
        return x


class ConvEmbedder(nn.Module):
    def __init__(self, in_channels: int, conv_channels: list[int], kernel_size: int = 2):
        super().__init__()
        layers = []
        for out_channels in conv_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.AdaptiveMaxPool1d(1))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.out_channels = conv_channels[-1]

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.amax(x, dim=2)
        return x


class HistEmbedNet(nn.Module):
    def __init__(
            self,
            in_features: int,
            num_years_back: int = 3,
            row_embed_layers: Optional[list[int]] = None,
            conv_channels: Optional[list[int]] = None,
            head_layers: Optional[List[int]] = None,
            **kwargs
    ):
        """
        param in_features: number of yearly features
        param in_features: number of years of history
        param row_embed_layers: list that specified the year-embedding MLP layers dimensions
        param conv_channels: list that specified the year-embedding conv layers dimensions
        param row_embed_processor: optional module to process the output of row_embed module
        param head_layers: list that specified the head MLP layers dimensions
        """
        super().__init__()
        if row_embed_layers is None:
            row_embed_layers = [64, 16, 8]
        if conv_channels is None:
            conv_channels = [32, 16]
        if head_layers is None:
            head_layers = [64, 32, 16, 8]

        self.in_features = in_features
        self.row_embed_layers: List[int] = row_embed_layers
        self.conv_channels: List[int] = conv_channels
        self.head_layers: List[int] = head_layers
        self.input_mode = '2d'

        feature_per_year = in_features // num_years_back
        self.row_embedder = RowEmbedder(feature_per_year, self.row_embed_layers)
        dim_row_embed = self.row_embedder.out_dim
        self.row_norm = nn.LayerNorm(dim_row_embed)
        self.row_processor_output_dim = num_years_back * dim_row_embed
        self.conv_embedder = ConvEmbedder(feature_per_year, conv_channels=self.conv_channels, kernel_size=3)
        self.dim_conv_embed = self.conv_channels[-1]

        # Head
        self.head_input_dim = self.row_processor_output_dim + self.dim_conv_embed
        head_net = []
        in_dim = self.head_input_dim
        for out_dim in self.head_layers:
            head_net.append(nn.Linear(in_dim, out_dim))
            head_net.append(nn.ReLU())
            in_dim = out_dim
        self.final_linear = nn.Linear(in_dim, 2)  # binary logits
        self.head = nn.Sequential(*head_net)

    @property
    def network_class_name(self) -> str:
        return (f"StatPredictDLModel"
                f"-emb{'-'.join([str(x) for x in self.row_embed_layers])}"
                f"-{'-'.join([str(x) for x in self.head_layers])}")

    def forward(self, x):
        # x shape: (batch, num_years_back, features)
        row_embeds = self.row_embedder(x)
        row_vec = row_embeds.flatten(start_dim=1)
        conv_vec = self.conv_embedder(x)
        combined = torch.cat([row_vec, conv_vec], dim=1)
        x_head = self.head(combined)
        return self.final_linear(x_head)


networks_to_class = {
    'MLPNet': MLPNet,
    'HistEmbedNet': HistEmbedNet,
}


class TrainingArgs:
    def __init__(self, **kwargs):
        self.num_epochs = kwargs.get('num_epochs', DefaultParameters.num_epochs)
        self.early_stop_patience = kwargs.get('early_stop_patience', DefaultParameters.early_stop)
        self.plot_loss = kwargs.get('plot_loss', True)
        self.batch_size = kwargs.get('batch_size', DefaultParameters.batch_size)
        self.optimizer_cls = kwargs.get('optimizer', torch.optim.AdamW)
        self.optimizer_args = dict(lr=kwargs.get('lr', DefaultParameters.learning_rate),
                                   weight_decay=kwargs.get('weight_decay', DefaultParameters.weight_decay),
                                   betas=kwargs.get('betas', DefaultParameters.betas),
                                   eps=kwargs.get('eps', DefaultParameters.eps)
                                   )


class DLArgs(ModelArgs):
    def __init__(self, apply_feature_selection: bool = False, **kwargs):
        if 'filter_feature_families' not in kwargs:
            kwargs['filter_feature_families'] = []
        super().__init__(apply_feature_selection=apply_feature_selection, **kwargs)
        self.args = dict()
        self.training_args: TrainingArgs = TrainingArgs(**kwargs)
        self.cls_threshold = kwargs.get('cls_threshold', 0.5)
        self.numeric_features_only: bool = True


class HistEmbedNetModelArgs(DLArgs):
    def __init__(self,
                 row_embed_layers: Optional[list[int]] = None,
                 conv_channels: Optional[list[int]] = None,
                 row_embed_processor: bool = True,
                 head_layers: Optional[List[int]] = None,
                 conv_kernel: Optional[int] = 3,
                 **kwargs):
        super().__init__(**kwargs)
        self.class_weight = None
        self.args = dict(row_embed_layers=row_embed_layers,
                         conv_channels=conv_channels,
                         row_embed_processor=row_embed_processor,
                         head_layers=head_layers,
                         conv_kernel=conv_kernel)


class DLStatPredict(MLStatPredict):
    def __init__(self, model_args: DLArgs, **kwargs):
        super().__init__(model_args, **kwargs)
        self.model_args = model_args
        self.scaler: Optional[Any] = kwargs.get('scaler', StandardScaler())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = None
        self.model = None
        self.cls_threshold = self.model_args.cls_threshold
        self.compute_features_importance = False

    def get_model_name(self, kwargs) -> str:
        if kwargs.get('name', None) is not None:
            return kwargs['name']
        else:
            return (f"{self.model_args.model_class_name}_"
                    f"{self.min_years_back}-{self.num_years_back}yback_{self.model_suffix}")

    @property
    def model_suffix(self) -> str:
        suffix = ''
        if self.model_args.filter_feature_families is not None:
            for fam in self.model_args.filter_feature_families:
                suffix += f'-{fam[:3]}'
        suffix += f"_lr-{round(self.model_args.training_args.optimizer_args['lr'], 5)}"
        if self.career_phase_model is not None:
            phase_str = self.model_args.career_phase_model_path.split('-by-')[1].split('.pickle')[0] \
                .replace('_series-raw', '')
            suffix += f"_SplitPhaseBy-{phase_str}"
        else:
            suffix += '_wo_phase_model'
        return suffix

    @property
    def batch_size(self):
        return self.model_args.training_args.batch_size

    @property
    def num_epochs(self):
        return self.model_args.training_args.num_epochs

    def get_optimizer(self):
        return self.model_args.training_args.optimizer_cls(self.model.parameters(),
                                                           **self.model_args.training_args.optimizer_args)

    @property
    def weights_path(self):
        return self.model_path.replace('pickle', 'pt')

    def save_model(self):
        super().save_model()
        torch.save(self.model.state_dict(), self.weights_path)

    def load_model(self):
        super().load_model()
        print(f'\nLoading {self.name} from {self.model_path}...')
        self.model.load_state_dict(torch.load(self.weights_path, weights_only=True))
        self.model.eval()

    def init_model(self, params: Dict = None, **kwargs):
        y: Optional[pd.Series] = kwargs.get('labels', None)
        self.model = networks_to_class[self.model_args.model_class_name](self.model_input_dim,
                                                                         num_years_back=self.num_years_back,
                                                                         **self.model_args.args,
                                                                         **kwargs)
        print(f'{self.model_args.model_class_name} architecture description:')
        print(self.model)

        if y is not None:
            weights = compute_balanced_class_weights(y)
            self.model_args.class_weight = weights
        else:
            weights = torch.from_numpy(np.array([1, 1])).flatten()
        self.loss = nn.CrossEntropyLoss(weights)

    def convert_to_torch_dataset(self, x: pd.DataFrame, y: List = None) -> Dataset:
        # Create features dataset
        x = torch.from_numpy(x.fillna(-1).values).to(self.device)
        if self.model.input_mode == '2d':
            # Transform feature vector into 2d: num years x features per year
            feature_per_year = int(x.shape[1] / self.num_years_back)
            x = x.view(-1, self.num_years_back, feature_per_year)
        elif self.model.input_mode != '1d':
            raise ValueError(f"Unfamiliar DL model.input_mode: {self.model.input_mode} (allowed values = '1d'/'2d')")

        if y is not None:
            y = torch.from_numpy(np.array(y)).long().to(self.device)
            return TensorDataset(x, y)
        return TensorDataset(x)

    def validation_iteration(self,
                             val_dl: DataLoader,
                             val_stats: dict,
                             epoch: int,
                             best_model_state_dict: Optional[Dict]
                             ) -> (dict, dict, bool):
        """
        Validation iteration
        return:

        """
        self.model.eval()
        with torch.no_grad():
            val_loss_epoch = []
            for x, y in val_dl:
                y_val_pred = self.model(x.float())
                val_loss_epoch.append(self.loss(y_val_pred.float(), y).item())
            val_stats['val_loss'].append(np.mean(val_loss_epoch))
            # Check if we improved the loss or not
            if val_stats['val_loss'][-1] < val_stats['best_val_loss']:
                val_stats['best_val_loss'] = val_stats['val_loss'][-1]
                best_model_state_dict = self.model.state_dict()
            elif epoch - np.argmin(val_stats['val_loss']) > self.model_args.training_args.early_stop_patience:
                print('Reached early stop > BREAK')
                return val_stats, best_model_state_dict, True
        return val_stats, best_model_state_dict, False

    def _fit(self,
             x_train: pd.DataFrame,
             y_train: np.array,
             x_val: pd.DataFrame,
             y_val: np.array,
             params: Dict = None):
        """ Pytorch model training loop """
        self.model_input_dim = x_train.shape[1]
        self.init_model(labels=y_train)

        train_stats = dict(train_loss=[])
        val_stats = dict(best_val_loss=float('inf'), val_loss=[])
        optimizer = self.get_optimizer()
        # Torch Datasets
        train_ds = self.convert_to_torch_dataset(x_train, y_train)
        val_ds = self.convert_to_torch_dataset(x_val, y_val)
        # Dataloaders
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)
        best_model_state_dict = None
        for epoch in tqdm(range(self.num_epochs), desc='epoch'):
            self.model.train()
            train_loss_epoch = []
            for x, y in train_dl:
                optimizer.zero_grad()
                y_pred = self.model(x.float())
                assert y.dim() == 1
                loss = self.loss(y_pred.float(), y)
                loss.backward()
                optimizer.step()
                train_loss_epoch.append(float(loss.item()))
            train_stats['train_loss'].append(np.mean(train_loss_epoch))
            val_stats, best_model_state_dict, stop_criteria = (
                self.validation_iteration(val_dl, val_stats, epoch, best_model_state_dict)
            )
            if stop_criteria:
                train_stats['training_epochs'] = epoch + 1  # count starts at 0
                break
            print(
                f"Epoch {epoch + 1}/{self.num_epochs}:\n - "
                f"Train / val loss {train_stats['train_loss'][-1]:.4f} / {val_stats['val_loss'][-1]:.4f}")

        loss_df = pd.DataFrame({
            'train': train_stats['train_loss'],
            'validation': val_stats['val_loss'],
            'epoch': list(range(len(train_stats['train_loss'])))
        })
        loss_fit = px.line(loss_df,
                           x='epoch', y=['train', 'validation'],
                           title=f'{self.name} Loss convergence over training')
        loss_fit.update_layout(template="plotly_white")
        loss_fit.write_html(os.path.join(self.reports_output_dir, f'{self.name}_training_loss_plot.html'))
        loss_fit.show()
        self.model.load_state_dict(best_model_state_dict)
        print(f"Best validation loss: {val_stats['best_val_loss']:.4f} "
              f"(epoch={np.array(val_stats['val_loss']).argmax()})")
        return train_stats['train_loss'], val_stats['val_loss']

    def add_phase_model_features(self,
                                 dataset: StatsDataset,
                                 x_train: pd.DataFrame,
                                 x_val: pd.DataFrame,
                                 x_test: pd.DataFrame):
        """
        Overrides parent method for 2d input extension
        """
        x_train, x_val, x_test = super().add_phase_model_features(dataset, x_train, x_val, x_test)
        phase_model_features = [c for c in x_train.columns if c.startswith(FeatureFamilies.phase_model)]

        def duplicate_over_years(df):
            for att in phase_model_features:
                for i in range(self.num_years_back):
                    new_col_name = f'{att}_row_{i}'
                    df[new_col_name] = df[att]
                    dataset.columns_mapping['numeric'].append(new_col_name)
                df = df.drop(columns=[att])
                if att in dataset.columns_mapping['numeric']:
                    dataset.columns_mapping['numeric'].remove(att)
            return df

        x_train = duplicate_over_years(x_train)
        x_val = duplicate_over_years(x_val)
        x_test = duplicate_over_years(x_test)
        return x_train, x_val, x_test

    def predict(self, x: pd.DataFrame) -> List[int]:
        probs = self.predict_proba(x)
        return [1 if p > self.cls_threshold else 0 for p in probs]

    def predict_proba(self, x: pd.DataFrame) -> List[float]:
        self.model.eval()
        with torch.no_grad():
            ds = self.convert_to_torch_dataset(x)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            preds = []
            for xb in dl:
                logits = self.model(xb[0].float())
                probs = torch.softmax(logits, dim=1)
                preds.extend(probs[:, 1].tolist())
        return preds
