import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(self.dropout(gru_out[:, -1, :]))
        return out


def forecast(model, data, sequence_length, steps_ahead):
    model.eval()

    data = np.array(data)

    current_sequence = data[-sequence_length:].reshape(1, sequence_length, -1)
    predictions = []

    with torch.no_grad():
        for _ in range(steps_ahead):
            input_seq = torch.tensor(current_sequence, dtype=torch.float32)

            output = model(input_seq)
            predicted_value = torch.sigmoid(output).item()

            predicted_class = 1 if predicted_value >= 0.5 else 0

            predictions.append(predicted_class)

            new_sequence = np.append(current_sequence[0, 1:, :],
                                     [[predicted_class] * current_sequence.shape[2]], axis=0)
            current_sequence = new_sequence.reshape(1, sequence_length, -1)

            movement_interpretation = ['Increase' if pred == 1 else 'Decrease' for pred in predictions]

    return movement_interpretation

def prepare_data_for_forecast(df):
    sorted_df = df.sort_values(by='id')

    high_prices = sorted_df.loc[:, 'high'].values
    low_prices = sorted_df.loc[:, 'low'].values
    mid_prices = (high_prices + low_prices) / 2.0

    mid_price_changes = np.diff(mid_prices) / mid_prices[:-1] * 100
    mid_price_changes = np.insert(mid_price_changes, 0, 0)

    features = sorted_df[
        ['volume', 'ma7', 'ma21', 'bollinger_upper', 'bollinger_lower', 'volatility',
         'close_usd_index', 'close_oil', 'close_gold', 'hash_rate']].values
    feature_changes = np.diff(features, axis=0) / features[:-1] * 100
    feature_changes = np.insert(feature_changes, 0, 0, axis=0)

    combined_features = np.column_stack((mid_price_changes.reshape(-1, 1), feature_changes))

    sequence_length = 100
    sequence_data = []
    sequence_labels = []

    for i in range(len(combined_features) - sequence_length):
        sequence_data.append(combined_features[i:i + sequence_length])
        # Labels based on whether the next mid_price_change is positive (1) or negative (0)
        sequence_labels.append(1 if mid_price_changes[i + sequence_length] > 0 else 0)

    sequence_data = np.array(sequence_data)
    sequence_labels = np.array(sequence_labels)


    split_index = int(len(sequence_data) * 0.8)
    train_data = sequence_data[:split_index]
    train_labels = sequence_labels[:split_index]
    test_data = sequence_data[split_index:]
    test_labels = sequence_labels[split_index:]

    train_data = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    input_size = combined_features.shape[1]

    all_data = np.concatenate(
        (train_data.tensors[0].numpy(), test_data.tensors[0].numpy())).reshape(-1, input_size)

    return all_data