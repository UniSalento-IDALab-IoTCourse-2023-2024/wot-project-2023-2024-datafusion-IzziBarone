import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# https://github.com/lemmingapex/trilateration

class TrilaterationFunction:
    def __init__(self, positions, distances):
        """
        :param positions: Matrice delle posizioni degli access point e dei beacon, forma (n, 2).
        :param distances: Vettore delle distanze dai punti noti, forma (n,).
        """
        if len(positions) < 2:
            raise ValueError("Serve almeno due posizioni.")
        if len(positions) != len(distances):
            raise ValueError("Il numero delle posizioni non corrisponde al numero delle distanze.")
        self.positions = np.array(positions)
        self.distances = np.array(distances)
        self.epsilon = 1e-7

    def residuals(self, point):
        """
        Calcola i residui, che sono la differenza tra la distanza calcolata e la distanza misurata.
        :param point: Le coordinate del punto da stimare, forma (2,).
        :return: Vettore dei residui, forma (n,).
        """
        return np.sqrt(np.sum((self.positions - point) ** 2, axis=1)) - self.distances


def trilateration(positions, distances):
    """
    Calcola la posizione (x, y) data dalle coordinate di tre punti noti e le distanze da ciascun punto.
    :param positions: Matrice delle posizioni degli access point e dei beacon, forma (n, 2).
    :param distances: Vettore delle distanze dai punti noti, forma (n,).
    :return: Coordinate stimate (x, y).
    """
    trilateration_func = TrilaterationFunction(positions, distances)
    initial_guess = np.mean(positions, axis=0)  # Usa la media delle posizioni come stima iniziale
    result = least_squares(trilateration_func.residuals, initial_guess)
    return result.x


def trilateration_classic(p1, p2, p3, d1, d2, d3):
    """
    Calcola la posizione (x, y) data dalle coordinate di tre punti noti e le distanze da ciascun punto.
    :param p1: Coordinate del primo punto (x1, y1)
    :param p2: Coordinate del secondo punto (x2, y2)
    :param p3: Coordinate del terzo punto (x3, y3)
    :param d1: Distanza dal primo punto
    :param d2: Distanza dal secondo punto
    :param d3: Distanza dal terzo punto
    :return: Coordinate stimate (x, y)
    """
    A = np.array([
        [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
        [2 * (p3[0] - p1[0]), 2 * (p3[1] - p1[1])]
    ])

    B = np.array([
        d1 ** 2 - d2 ** 2 - p1[0] ** 2 + p2[0] ** 2 - p1[1] ** 2 + p2[1] ** 2,
        d1 ** 2 - d3 ** 2 - p1[0] ** 2 + p3[0] ** 2 - p1[1] ** 2 + p3[1] ** 2
    ])

    pos = np.linalg.solve(A, B)
    return pos


def select_best(row, beacon_columns, beacon_positions):
    d_values = row[beacon_columns]
    top_beacons = d_values.nsmallest(3).index

    top_positions = [beacon_positions[beacon] for beacon in top_beacons]
    top_distances = [row[beacon] for beacon in top_beacons]

    return top_positions, top_distances


data = pd.read_csv('../data/distances/combined_fingerprints_distance.csv')
beacon_data = pd.read_csv("../data/ble_devices.csv")

beacon_positions = {row['ID']: (row['X'], row['Y']) for index, row in beacon_data.iterrows()}
print(beacon_positions)

beacon_columns = [col for col in data.columns if col.startswith('BEACON_')]
print(beacon_columns)

positions = []
for index, row in data.iterrows():
    position, distances = select_best(row, beacon_columns, beacon_positions)
    estimate = trilateration(position, distances)
    positions.append((row["TIMESTAMP"], estimate[0], estimate[1]))

pos_df = pd.DataFrame(positions, columns=['TIMESTAMP', 'x', 'y'])
pos_df['TIMESTAMP'] = pos_df['TIMESTAMP'].astype(int)
pos_df.to_csv('../data/positions/estimated_positions.csv', index=False)

#real_positions = pd.read_csv('../data/wlan_fingerprints.csv')
real_positions = pd.read_csv('../data/aggregated/wlan-ble-imu_fingerprints.csv')
real_positions = real_positions.iloc[:, [0, 1, 2]]
merged = pd.merge(pos_df, real_positions, on='TIMESTAMP')

merged.to_csv('../data/positions/merged_positions.csv', index=False)
