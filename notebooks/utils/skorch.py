# coding=utf-8
#
# The copyright of this file belongs to anonymized. The file cannot be
# reproduced in whole or in part, stored in a retrieval system,
# transmitted in any form, or by any means electronic, mechanical,
# photocopying, or otherwise, without the prior permission of the owner.
#
# (c) 2020 anonymized, Strictly Confidential
"""Wrappers for using NNs with scikit-learn API (based on skorch).

TODO
 - checkpoint best epoch iteration based on train_loss (with a callback?);
 - use the best iteration for predictions;

"""
import random
from functools import partial
from typing import Tuple, List, Callable, Optional, Union

import pandas
import numpy as np
import skorch
import torch
from torch import nn
from torch.nn import functional as F

from .fairautoml_utils_classpath import import_object


class InputShapeSetter(skorch.callbacks.Callback):
    """Skorch callback for automatically setting the input dimension.
    """

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        net.set_params(module__input_dim=X.shape[-1])


class FeedForwardClassifier(skorch.NeuralNetBinaryClassifier):
    """Skorch wrapper for a Feed Forward Neural Network.

    TODO
     - this class can be abstracted to serve any Neural Network architecture,
     by passing the classpath for the pytorch module as an argument instead of
     having the FFNN import hardcoded.
    """

    MODEL_ARGS_KEY = 'model__'

    def __init__(
            self,
            criterion: Union[type, str] = nn.BCELoss,
            optimizer: Union[type, str] = torch.optim.Adam,
            use_cuda: bool = True,
            random_state: int = 42,
            n_jobs: int = -1,
            **kwargs,
        ):
        if use_cuda and torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self.random_state = random_state
        self.n_jobs = n_jobs    # NOTE: currently being ignored

        # Load types if passed as a string (this enables choosing from the yaml)
        if isinstance(criterion, str):
            criterion = import_object(criterion)

        if isinstance(optimizer, str):
            optimizer = import_object(optimizer)

        # Parse key-word arguments
        self.model_kwargs = {
            k[len(self.MODEL_ARGS_KEY):]: v
            for k, v in kwargs.items() if k.startswith(self.MODEL_ARGS_KEY)
        }
        self.skorch_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(self.MODEL_ARGS_KEY)
        }

        super().__init__(
            partial(FFNN, **self.model_kwargs),
            # callbacks=[InputShapeSetter()],
            device=self._device,
            train_split=None,
            criterion=criterion,
            optimizer=optimizer,
            **self.skorch_kwargs,
        )

    @staticmethod
    def seed(seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def convert_input_data(data) -> np.ndarray:
        if isinstance(data, (pandas.DataFrame, pandas.Series)):
            return data.to_numpy().astype(np.float32)
        return data

    def initialize(self):
        self.seed(self.random_state)
        super().initialize()

    def fit(self, X, y, **fit_params):
        X, y = map(self.convert_input_data, [X, y])
        return super().fit(X, y, **fit_params)

    def predict_proba(self, X):
        X = self.convert_input_data(X)
        return super().predict_proba(X)


class FFNN(nn.Module):
    """Generic Feed Forward Neural Network
    https://github.com/AndreFCruz/bias-detection/blob/master/src/architectures/ffnn_architectures.py
    """

    def __init__(
            self, input_dim: int,
            output_dim: int = 1,
            hidden_layers: List[int] = None,
            dropout: Optional[float] = None,
            activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
            use_batch_norm: bool = False,
        ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = list()
        assert input_dim > 0 and output_dim > 0

        self.activ_function = activation
        self.output_activ_function = torch.sigmoid if output_dim == 1 else F.softmax

        self.linear_layers, self.bn_layers = FFNN.make_linear_layers(
            input_dim, output_dim, hidden_layers, use_batch_norm,
        )

        if dropout is None:
            # No dropout
            self.dropout_layers = None
        else:
            # Same dropout on all layers
            self.dropout_layers = nn.ModuleList(
                [nn.Dropout(p=dropout) for _ in range(len(hidden_layers))]
            )

        assert (len(self.linear_layers) - 1) == len(hidden_layers)
        assert self.bn_layers is None or len(self.bn_layers) == len(hidden_layers)
        assert self.dropout_layers is None or len(self.dropout_layers) == len(hidden_layers)

    @staticmethod
    def make_linear_layers(
            input_size: int,
            output_size: int,
            hidden_layers: List[int],
            use_batch_norm: bool = False,
        ) -> Tuple[nn.ModuleList, Optional[nn.ModuleList]]:
        """Dynamically construct linear layers.

        Parameters
        ----------
        input_size : int
            The (single-dimensional) shape of the input data.
            The number of neurons in the input layer.
        output_size : int
            The number of neurons in the output layer.
        hidden_layers : List[int]
            The number of neurons per hidden layer, ordered.
            E.g., [20, 10, 5] will generate three hidden layers, respectively
            with 20 -> 10 -> 5 neurons, plus the input and output layers.
        use_batch_norm : bool
            Whether to use batch normalization after all layers.

        Returns
        -------
        A Module corresponding to the constructed Linear layers, as well as a
        second module corresponding to the BatchNorm layers (if applicable).
        """
        if len(hidden_layers) == 0:
            return [], []

        linear_layers = list()
        bn_layers = list()
        for i, out_neurons in enumerate(hidden_layers):
            in_neurons = input_size if i == 0 else hidden_layers[i-1]

            if use_batch_norm:
                linear_layers.append(nn.Linear(in_neurons, out_neurons, bias=False))
                bn_layers.append(nn.BatchNorm1d(out_neurons))
            else:
                linear_layers.append(nn.Linear(in_neurons, out_neurons, bias=True))

        # Output layer
        linear_layers.append(nn.Linear(hidden_layers[-1], output_size, bias=True))

        # NOTE: needs to be wrapped on a ModuleList so the parameters can be found by the optimizer
        return \
            nn.ModuleList(linear_layers), \
            nn.ModuleList(bn_layers) if use_batch_norm else None

    def forward(self, x):
        for idx, lin_layer in enumerate(self.linear_layers):
            x = lin_layer(x)
            if idx == len(self.linear_layers) - 1:
                continue

            if self.bn_layers is not None and idx < len(self.bn_layers):
                x = self.bn_layers[idx](x)

            x = self.activ_function(x)
            if self.dropout_layers is not None and idx < len(self.dropout_layers):
                x = self.dropout_layers[idx](x)

        return self.output_activ_function(x)
