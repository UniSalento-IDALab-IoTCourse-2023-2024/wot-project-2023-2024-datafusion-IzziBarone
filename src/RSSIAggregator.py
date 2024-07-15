import pandas as pd
import numpy as np


def preprocess_data(data):
    data.iloc[:, 4:] = data.iloc[:, 4:].astype(np.int64)
    # Replace 100 with -1000 (terrible signal)
    data.iloc[:, 4:] = data.iloc[:, 4:].replace(100, np.nan)
    # Passare dataframe con solo prima colonna e dalla 4 in poi
    data = data.iloc[:, [0] + list(range(4, data.shape[1]))]
    return data


class RSSIAggregator:
    def __init__(self, wifi_data, ble_data, time_col):
        self.wifi_data = wifi_data
        self.ble_data = ble_data
        self._time_col = time_col

    @staticmethod
    def _handle_null_values(data):
        """
        Gestisce i valori nulli nei dati usando l'interpolazione lineare.
        :param data: DataFrame con i dati RSSI.
        :return: DataFrame con i valori nulli gestiti.
        """
        data.interpolate(method='linear', inplace=True, limit_direction='both')
        return data

    def _define_intervals(self):
        # Convertire i timestamp in millisecondi
        self.wifi_data[self._time_col] = self.wifi_data[self._time_col].astype(int)
        self.ble_data[self._time_col] = self.ble_data[self._time_col].astype(int)

        # Definire gli intervalli di tempo basati sui timestamp del Wi-Fi
        intervals = [(self.wifi_data[self._time_col].iloc[i], self.wifi_data[self._time_col].iloc[i + 1])
                     for i in range(len(self.wifi_data[self._time_col]) - 1)]
        intervals.insert(0, (self.wifi_data[self._time_col].iloc[0] - 0.5, self.wifi_data[self._time_col].iloc[0]))
        return intervals

    def _aggregate_rssi(self, data, intervals):
        aggregated_data = []
        rssi_columns = [col for col in data.columns if col != self._time_col]

        for start, end in intervals:
            mask = (data[self._time_col] > start) & (data[self._time_col] <= end)
            aggregated_row = [end]

            for col in rssi_columns:
                if mask.any():
                    mean_rssi = data.loc[mask, col].mean()
                else:
                    mean_rssi = np.nan
                aggregated_row.append(mean_rssi)

            aggregated_data.append(aggregated_row)

        columns = [self._time_col] + rssi_columns
        return pd.DataFrame(aggregated_data, columns=columns)

    @staticmethod
    def _handle_aggregated_nulls(combined_data):
        """
        Gestisce i valori nulli nei dati aggregati BLE.
        :param combined_data: DataFrame combinato Wi-Fi e BLE con dati aggregati.
        :return: DataFrame con valori nulli nelle colonne BLE gestiti.
        """
        combined_data.interpolate(method='linear', inplace=True, limit_direction='both')
        return combined_data

    def aggregate(self):
        self.wifi_data = self._handle_null_values(self.wifi_data)
        self.ble_data = self._handle_null_values(self.ble_data)

        intervals = self._define_intervals()
        aggregated_ble_data = self._aggregate_rssi(self.ble_data, intervals)
        combined_data = pd.merge(self.wifi_data, aggregated_ble_data, on=self._time_col, how='left')

        # Elimina le righe con tutti i valori nulli tranne il timestamp
        combined_data.dropna(how='all', subset=[col for col in combined_data.columns if col != self._time_col],
                             inplace=True)

        # Gestire i valori nulli nei dati aggregati BLE
        combined_data = self._handle_aggregated_nulls(combined_data)

        return combined_data


# Esempio di utilizzo della classe

wifi_data = pd.read_csv('../data/wlan_fingerprints.csv')
wifi_data = preprocess_data(wifi_data)

ble_data = pd.read_csv('../data/ble_fingerprints.csv')
ble_data = preprocess_data(ble_data)

aggregator = RSSIAggregator(wifi_data, ble_data, "TIMESTAMP")
combined_data = aggregator.aggregate()

combined_data.to_csv('../data/aggregated/combined_fingerprints.csv', index=False)
