import os
import sys
import warnings
from typing import Callable
import math

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
import torch as T
import torch.optim as optim
from torch.nn import Flatten, Conv1d, ReLU, Linear, Module, MSELoss
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device used: ", device)

PREDICTOR_FILE_NAME = "predictor.joblib"
MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


def get_activation(activation: str) -> Callable:
    """
    Return the activation function based on the input string.

    This function returns a callable activation function from the
    torch.nn.functional package.

    Args:
        activation (str): Name of the activation function.

    Returns:
        Callable: The requested activation function. If 'none' is specified,
        it will return an identity function.

    Raises:
        Exception: If the activation string does not match any known
        activation functions ('relu', 'tanh', or 'none').

    """
    if activation == "tanh":
        return F.tanh
    elif activation == "relu":
        return F.relu
    elif activation == "none":
        return lambda x: x  # Identity function, doesn't change input
    else:
        raise ValueError(
            f"Error: Unrecognized activation type: {activation}. "
            "Must be one of ['relu', 'tanh', 'none']."
        )


def get_patience_factor(N):
    # magic number - just picked through trial and error
    if N < 100:
        return 30
    patience = int(37 - math.log(N, 1.5))
    return patience


def get_loss(model, device, data_loader, loss_function):
    model.eval()
    loss_total = 0
    with torch.no_grad():
        for data in data_loader:
            X, y = data[0].to(device), data[1].to(device)
            output = model(X)
            loss = loss_function(y, output)
            loss_total += loss.item()
    return loss_total / len(data_loader)


class CustomDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Net(Module):
    def __init__(self, feat_dim, encode_len, decode_len, activation):
        super(Net, self).__init__()
        self.feat_dim = feat_dim
        self.encode_length = encode_len
        self.decode_len = decode_len
        self.activation = get_activation(activation)

        dim1 = 100
        dim2 = 50
        dim3 = 25

        self.conv1 = Conv1d(
            in_channels=self.feat_dim,
            out_channels=dim1,
            kernel_size=4,
            stride=1,
            padding="same",
        )
        self.conv2 = Conv1d(
            in_channels=dim1, out_channels=dim2, kernel_size=8, stride=1, padding="same"
        )
        self.conv3 = Conv1d(
            in_channels=dim2,
            out_channels=dim3,
            kernel_size=16,
            stride=1,
            padding="same",
        )
        self.fc = Linear(
            in_features=dim3 * min(self.decode_len, self.encode_length),
            out_features=self.decode_len,
        )
        self.flatten = Flatten()

    def forward(self, X):
        x = X.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = x[:, -self.decode_len :, :]
        x = self.flatten(x)
        x = self.activation(x)
        x = self.fc(x)
        out = x
        return out

    def get_num_parameters(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp


class Forecaster:
    """CNN Timeseries Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    MODEL_NAME = "CNN_Timeseries_Forecaster"

    def __init__(
        self, encode_len: int, decode_len: int, feat_dim: int, activation: str, **kwargs
    ):
        """Construct a new CNN Forecaster."""
        self.encode_len = encode_len
        self.decode_len = decode_len
        self.feat_dim = feat_dim
        self.activation = activation
        self.batch_size = 64

        self.net = Net(
            feat_dim=self.feat_dim,
            encode_len=self.encode_len,
            decode_len=self.decode_len,
            activation=self.activation,
        )

        self.net.to(device)
        # print(self.net.get_num_parameters()) ; sys.exit()
        self.criterion = MSELoss()
        self.optimizer = optim.Adam(self.net.parameters())
        self.print_period = 1

    def _get_X_and_y(self, data: np.ndarray, is_train: bool = True) -> np.ndarray:
        """Extract X (historical target series), y (forecast window target)
        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape
        if D != self.feat_dim:
            raise ValueError(
                f"Training data expected to have {self.feat_dim} feature dim. "
                f"Found {D}"
            )
        if is_train:
            if T != self.encode_len + self.decode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len + self.decode_len}"
                    f" length on axis 1. Found length {T}"
                )
            X = data[:, : self.encode_len, :]
            y = data[:, self.encode_len :, 0]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            X = data[:, -self.encode_len :, :]
            y = None
        return X, y

    def fit(self, train_data, valid_data, max_epochs=100, verbose=1):

        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        if valid_data is not None:
            valid_X, valid_y = self._get_X_and_y(valid_data, is_train=True)
        else:
            valid_X, valid_y = None, None

        # print(train_X.shape, train_y.shape)
        # if valid_X is not None: print(valid_X.shape, valid_y.shape)

        patience = get_patience_factor(train_X.shape[0])
        # print(f"{patience=}")

        train_X, train_y = torch.FloatTensor(train_X), torch.FloatTensor(train_y)
        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=int(self.batch_size), shuffle=True
        )

        if valid_X is not None and valid_y is not None:
            valid_X, valid_y = torch.FloatTensor(valid_X), torch.FloatTensor(valid_y)
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=int(self.batch_size), shuffle=True
            )
        else:
            valid_loader = None

        losses = self._run_training(
            train_loader,
            valid_loader,
            max_epochs,
            use_early_stopping=True,
            patience=patience,
            verbose=verbose,
        )
        return losses

    def _run_training(
        self,
        train_loader,
        valid_loader,
        max_epochs,
        use_early_stopping=True,
        patience=10,
        verbose=1,
    ):

        best_loss = 1e7
        losses = []
        min_epochs = 10
        for epoch in range(max_epochs):
            self.net.train()
            for data in train_loader:
                X, y = data[0].to(device), data[1].to(device)
                # sys.exit()
                # print(inputs); sys.exit()
                # Feed Forward
                preds = self.net(X)
                # Loss Calculation
                loss = self.criterion(y, preds)
                # Clear the gradient buffer (we don't want to accumulate gradients)
                self.optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Weight Update: w <-- w - lr * gradient
                self.optimizer.step()

            current_loss = loss.item()

            if use_early_stopping:
                # Early stopping
                if valid_loader is not None:
                    current_loss = get_loss(
                        self.net, device, valid_loader, self.criterion
                    )
                losses.append({"epoch": epoch, "loss": current_loss})
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and epoch >= min_epochs:
                        if verbose == 1:
                            print(f"Early stopping after {epoch=}!")
                        return losses

            else:
                losses.append({"epoch": epoch, "loss": current_loss})
            # Show progress
            if verbose == 1:
                if epoch % self.print_period == 0 or epoch == max_epochs - 1:
                    print(
                        f"Epoch: {epoch+1}/{max_epochs}, loss: {np.round(current_loss, 5)}"
                    )

        return losses

    def predict(self, data):
        X = self._get_X_and_y(data, is_train=False)[0]
        pred_X = torch.FloatTensor(X)
        # Initialize dataset and dataloader with only X
        pred_dataset = CustomDataset(pred_X)
        pred_loader = DataLoader(
            dataset=pred_dataset, batch_size=int(self.batch_size), shuffle=False
        )

        all_preds = []
        for data in pred_loader:
            # Get X and send it to the device
            X = data.to(device)
            preds = self.net(X).detach().cpu().numpy()
            preds = preds[:, -self.decode_len :]
            all_preds.append(preds)

        preds = np.concatenate(all_preds, axis=0)
        preds = np.expand_dims(preds, axis=-1)
        return preds

    def summary(self):
        self.model.summary()

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_and_y(test_data, is_train=True)
        if self.net is not None:
            x_test, y_test = torch.FloatTensor(x_test), torch.FloatTensor(y_test)
            dataset = CustomDataset(x_test, y_test)
            data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
            current_loss = get_loss(self.net, device, data_loader, self.criterion)
            return current_loss

    def save(self, model_path):
        model_params = {
            "encode_len": self.encode_len,
            "decode_len": self.decode_len,
            "feat_dim": self.feat_dim,
            "activation": self.activation,
        }
        joblib.dump(model_params, os.path.join(model_path, MODEL_PARAMS_FNAME))
        torch.save(self.net.state_dict(), os.path.join(model_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_path):
        model_params = joblib.load(os.path.join(model_path, MODEL_PARAMS_FNAME))
        classifier = cls(**model_params)
        classifier.net.load_state_dict(
            torch.load(os.path.join(model_path, MODEL_WTS_FNAME))
        )
        return classifier

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.MODEL_NAME}"


def train_predictor_model(
    history: pd.DataFrame,
    forecast_length: int,
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the forecaster model.

    Args:
        history (np.ndarray): The training data inputs.
        forecast_length (int): Length of forecast window.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        encode_len=history.shape[1] - forecast_length,
        decode_len=forecast_length,
        feat_dim=history.shape[2],
        **hyperparameters,
    )
    model.fit(
        train_data=history,
        valid_data=None,
    )
    return model


def predict_with_model(model: Forecaster, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (np.ndarray): The test input data for forecasting.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(test_data)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)


if __name__ == "__main__":

    N = 100
    T = 25
    D = 3

    model = Net(
        feat_dim=D,
        latent_dim=13,
        n_cnnlayers=2,
        decode_len=10,
        activation="relu",
    )
    model.to(device=device)

    X = torch.from_numpy(np.random.randn(N, T, D).astype(np.float32)).to(device)

    preds = model(X)
    print(preds.shape)
