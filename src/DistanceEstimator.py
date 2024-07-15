import pandas as pd

# 2.4Ghz
# params = (1.0, -40.0, 3.5, 2.5)

# 5Ghz wlan_ble_fingerprint.csv
params = (0.05, -45.0, 2.0, 2.5)


# wifi_rssi_test https://arxiv.org/pdf/2005.01877 https://github.com/pspachos/RSSI-Dataset-for-Indoor-Localization-Fingerprinting
#params = (0.5, -45.73, 2.162, 2.5)


class DistanceEstimator:
    def __init__(self, params):
        self.d_ref = params[0]  # reference distance
        self.power_ref = params[1]  # mean received power at reference distance
        self.path_loss_exp = params[2]  # path loss exponent
        self.stdev_power = params[3]  # standard deviation of received power
        self.uncertainty = 2 * self.stdev_power  # uncertainty in RSS corresponding to 95.45% confidence

    def get_distance(self, power_received):
        if power_received is None:
            return None
        d_est = self.d_ref * (10 ** (-(power_received - self.power_ref) / (10 * self.path_loss_exp)))
        return d_est

    def process_data(self, df, col_idx=2, threshold=10000):
        df.iloc[:, col_idx:] = df.iloc[:, col_idx:].map(lambda x: self.get_distance(x))
        df.iloc[:, col_idx:] = df.iloc[:, col_idx:].where(df.iloc[:, col_idx:] <= threshold)
        df.fillna(50, inplace=True)
        return df


# df = pd.read_csv('../data/wifi_rssi_test.csv')
# col_idx = 2
# df.iloc[:, col_idx:] = df.iloc[:, col_idx:].map(lambda x: get_distance(x))
#
# # valutare valore di threshold per le distanze
# # open space: 2.4 Ghz -> 45-90 m, 5 Ghz -> 20-45 m
# # indoor space: 2.4 Ghz -> 20-30 m, 5 Ghz -> 10-20 m
# threshold = 10000
# df.iloc[:, col_idx:] = df.iloc[:, col_idx:].where(df.iloc[:, col_idx:] <= threshold)
#
# df.to_csv('../data/distances/wlan_fingerprints_distance.csv', index=False)
#
# # d_est = d_ref * (10 ** (-(power_received - power_ref) / (10 * path_loss_exp)))
# # d_min = d_ref * (10 ** (-(power_received - power_ref + uncertainty) / (10 * path_loss_exp)))
# # d_max = d_ref * (10 ** (-(power_received - power_ref - uncertainty) / (10 * path_loss_exp)))

#df = pd.read_csv('../data/aggregated/combined_fingerprints.csv')
df = pd.read_csv('../data/aggregated/wlan-ble-imu_fingerprints.csv')
df.replace(100,-300,inplace=True)

col_idx = 3

estimator = DistanceEstimator(params)
estimation = estimator.process_data(df, col_idx)
estimation.to_csv('../data/distances/combined_fingerprints_distance.csv', index=False)
