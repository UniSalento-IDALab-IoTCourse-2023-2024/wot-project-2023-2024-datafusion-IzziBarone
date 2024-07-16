import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from src.IMUProcessor import IMUProcessor


# Step detection estimation, Pitch fusion and Heading estimation Based on 10.1109/ACCESS.2019.2891942

class IMUProcessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.acc_data = None
        self.gyro_data = None
        self.yaw = None

    def load_and_process_data(self):
        # Caricamento del dataset
        data = pd.read_csv(self.file_path)

        # Estrazione delle colonne necessarie
        self.acc_data = data[['TIMESTAMP', 'ACC_X', 'ACC_Y', 'ACC_Z']].copy(deep=True)
        self.gyro_data = data[['TIMESTAMP', 'RRATE_X', 'RRATE_Y', 'RRATE_Z']].copy(deep=True)
        self.yaw = data[['YAW']]

        # Costante di gravitazione
        g = 9.81

        # Calcolo di pitch (θ) e roll (φ) dall'accelerometro
        self.acc_data['PITCH'] = np.arcsin(self.acc_data['ACC_Y'] / g)
        self.acc_data['ROLL'] = np.arctan2(self.acc_data['ACC_X'], self.acc_data['ACC_Z'])
        self.acc_data['YAW'] = np.deg2rad(self.yaw)

        self.gyro_data['PITCH_GYRO'] = 0.0
        self.gyro_data['ROLL_GYRO'] = 0.0
        self.gyro_data['YAW_GYRO'] = 0.0

        # Inizializzazione dei valori di pitch e roll
        pitch_gyro = 0.0
        roll_gyro = 0.0
        yaw_gyro = 0.0

        # Conversione dei timestamp in secondi
        gyro_data = self.gyro_data.set_index('TIMESTAMP')

        # Integrazione delle velocità angolari per ottenere pitch e roll
        for i in range(1, len(gyro_data)):
            time_first = gyro_data.index[i - 1]
            time_second = gyro_data.index[i]
            diff = time_second - time_first
            dt = diff / 1000  # in seconds

            # Integrazione delle velocità angolari
            roll_gyro += self.gyro_data.iloc[i]['RRATE_X'] * dt
            pitch_gyro += self.gyro_data.iloc[i]['RRATE_Y'] * dt

            phi = roll_gyro
            theta = pitch_gyro
            omega_y = self.gyro_data.iloc[i]['RRATE_Y']
            omega_z = self.gyro_data.iloc[i]['RRATE_Z']

            yaw_gyro = (omega_y * np.sin(phi) / np.cos(theta) + omega_z * np.cos(phi) / np.cos(theta))

            # Salvataggio dei valori di pitch e roll
            self.gyro_data.iloc[i-1, self.gyro_data.columns.get_loc('PITCH_GYRO')] = pitch_gyro
            self.gyro_data.iloc[i-1, self.gyro_data.columns.get_loc('ROLL_GYRO')] = roll_gyro
            self.gyro_data.iloc[i-1, self.gyro_data.columns.get_loc('YAW_GYRO')] = yaw_gyro

        return self.acc_data, self.gyro_data

    def process_all(self):
        # Carica e processa i dati IMU
        self.load_and_process_data()

        # Applica il filtro di Kalman
        kalman_filter = KalmanFilterQuaternion(self.acc_data, self.gyro_data)
        filtered_data = kalman_filter.apply_filter()

        # Rileva i passi
        step_detector = StepDetector()
        steps = step_detector.detect_steps(
            pd.Series(filtered_data['PITCH_KALMAN'].values, index=filtered_data['TIMESTAMP']))

        # Calcola la lunghezza del passo
        stride_processor = StrideLengthProcessor(k=1.5)
        stride_lengths = stride_processor.process_stride_length(steps, filtered_data)

        # Processa l'orientamento
        heading_processor = HeadingProcessor(filtered_data, self.mag_data, self.gyro_data)
        heading_processor.calculate_heading_mag()
        heading_processor.calculate_heading_gyro()
        heading_kalman = heading_processor.filter_heading()

        # Restituisce un riepilogo dei risultati
        return {
            "filtered_data": filtered_data,
            "steps": steps,
            "stride_lengths": stride_lengths,
            "heading_kalman": heading_kalman
        }


class KalmanFilterQuaternion:
    def __init__(self, acc_data, gyro_data):
        self.acc_data = acc_data
        self.gyro_data = gyro_data
        self.x = np.array([1, 0, 0, 0])  # Stato iniziale [q0, q1, q2, q3]
        self.P = np.eye(4)  # Matrice di covarianza iniziale
        self.F = np.eye(4)  # Matrice di transizione di stato
        self.H = np.eye(4)  # Matrice di osservazione
        self.Q = 0.001 * np.eye(4)  # Covarianza del rumore di processo
        self.R = 0.0001 * np.eye(4)  # Covarianza del rumore di osservazione

    @staticmethod
    def get_B(omega):
        wx, wy, wz = omega
        return 0.5 * np.array([[0, -wx, -wy, -wz],
                               [wx, 0, wz, -wy],
                               [wy, -wz, 0, wx],
                               [wz, wy, -wx, 0]])

    def apply_filter(self):
        roll_kalman = []
        pitch_kalman = []

        for i in range(0, len(self.acc_data)):
            omega = self.gyro_data.iloc[i][['RRATE_X', 'RRATE_Y', 'RRATE_Z']].values
            B = self.get_B(omega)
            self.x = B @ self.x
            self.P = B @ self.P @ B.T + self.Q

            roll = self.acc_data.iloc[i]['ROLL']
            pitch = self.acc_data.iloc[i]['PITCH']
            yaw = self.acc_data.iloc[i]['YAW']
            #yaw = self.gyro_data.iloc[i]['YAW_GYRO']

            z = np.array([np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(
                pitch / 2) * np.sin(yaw / 2),
                          np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(
                              pitch / 2) * np.sin(yaw / 2),
                          np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(
                              pitch / 2) * np.sin(yaw / 2),
                          np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(
                              pitch / 2) * np.cos(yaw / 2)])

            # Aggiornamento
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = self.P - K @ self.H @ self.P

            # Calcolo di roll e pitch dal quaternione (https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles)
            # Normalizzazione del quaternione
            norm_x = self.x / np.linalg.norm(self.x)

            roll_x = np.arctan2(2 * (norm_x[0] * norm_x[1] + norm_x[2] * norm_x[3]),
                                1 - 2 * (norm_x[1] ** 2 + norm_x[2] ** 2))
            pitch_x = np.arcsin(2 * (norm_x[0] * norm_x[2] - norm_x[3] * norm_x[1]))
            #yaw_x = np.arctan2(2 * (norm_x[0] * norm_x[3] + norm_x[1] * norm_x[2]), 1 - 2 * (norm_x[2] ** 2 + norm_x[3] ** 2))

            # Salvataggio delle stime
            roll_kalman.append(roll_x)
            pitch_kalman.append(pitch_x)

        self.acc_data['ROLL_KALMAN'] = roll_kalman
        self.acc_data['PITCH_KALMAN'] = pitch_kalman

        return self.acc_data


class StepDetector:
    def __init__(self):
        pass

    @staticmethod
    def moving_average(data, window_size):
        return data.rolling(window=window_size, min_periods=1).mean()

    def detect_steps(self, pitch_data, min_interval=110, max_interval=500):

        window_size = 500
        pitch_moving_avg = self.moving_average(pitch_kalman_series, window_size)

        # Calcolo della soglia dinamica
        threshold_factor = 1
        dynamic_threshold = pitch_moving_avg * threshold_factor

        steps = []
        maxima = None
        minima = None
        last_max_index = -1
        last_min_index = -1

        for i in range(1, len(pitch_data) - 1):
            if pitch_data.iloc[i] > pitch_data.iloc[i - 1] and pitch_data.iloc[i] > pitch_data.iloc[i + 1]:
                if pitch_data.iloc[i] >= dynamic_threshold.iloc[i]:
                    maxima = pitch_data.iloc[i]
                    last_max_index = i

            if pitch_data.iloc[i] < pitch_data.iloc[i - 1] and pitch_data.iloc[i] < pitch_data.iloc[i + 1]:
                if pitch_data.iloc[i] < dynamic_threshold.iloc[i]:
                    minima = pitch_data.iloc[i]
                    last_min_index = i

            if maxima is not None and minima is not None:
                interval = (pitch_data.index[last_min_index] - pitch_data.index[last_max_index])
                if min_interval <= interval <= max_interval:
                    steps.append((last_max_index, last_min_index))
                    maxima = None
                    minima = None

        return steps


class StrideLengthProcessor:
    def __init__(self, k, stride_length_mean=0, q=0.2):
        self.k = k
        self.stride_length_mean = stride_length_mean
        self.q = q

    def process_stride_length(self, steps, data):
        timestamps = data['TIMESTAMP'].values

        lengths = []
        for i in range(0, len(steps)):
            start_idx = steps[i][0]  # Start index of the step
            end_idx = steps[i][1]  # End index of the step

            step_acc = data['ACC_U'].values[start_idx:end_idx + 1]

            if len(step_acc) == 0:
                continue

            # Usare il modello del paper per stimare la lunghezza del passo
            stride_length = self.k * (np.max(step_acc) - np.min(step_acc)) ** (1 / 4)

            # Filtraggio della lunghezza del passo
            if self.stride_length_mean != 0:
                if stride_length > self.stride_length_mean + self.q or stride_length < self.stride_length_mean - self.q:
                    stride_length = self.stride_length_mean

            lengths.append((timestamps[end_idx], stride_length))

        return lengths


class HeadingProcessor:
    def __init__(self, acc_data, mag_data, gyro_data):
        self.acc_data = acc_data
        self.mag_data = mag_data
        self.gyro_data = gyro_data

        self.heading_mag = None
        self.heading_gyro = None

        self.heading_kalman = []

        self.calculate_heading_mag()
        self.convert_to_euler()
        self.calculate_heading_gyro()

    def calculate_heading_mag(self):
        # Sfruttiamo roll e pitch calcolati dalla fusione dell'accelerometro con il giroscopio
        roll = self.acc_data['ROLL_KALMAN']
        pitch = self.acc_data['PITCH_KALMAN']

        Hx = np.zeros_like(roll)
        Hy = np.zeros_like(roll)
        Hz = np.zeros_like(roll)

        for i in range(len(roll)):
            phi = roll[i]
            theta = pitch[i]
            hx, hy, hz = self.mag_data.iloc[i]['MAG_X'], self.mag_data.iloc[i]['MAG_Y'], self.mag_data.iloc[i]['MAG_Z']

            # Matrice di rotazione
            R = np.array([
                [np.cos(phi), np.sin(phi) * np.sin(theta), -np.sin(phi) * np.cos(theta)],
                [0, np.cos(theta), np.sin(theta)],
                [np.sin(phi), -np.sin(theta) * np.cos(phi), np.cos(phi) * np.cos(theta)]
            ])

            # Vettore
            h = np.array([hx, hy, hz])

            # Prodotto tra la matrice di rotazione e il vettore del campo magnetico
            H = R @ h

            # Assegnazione delle componenti calcolate
            Hx[i] = H[0]
            Hy[i] = H[1]
            Hz[i] = H[2]

        self.mag_data['Hx'] = Hx
        self.mag_data['Hy'] = Hy
        self.mag_data['Hz'] = Hz

        self.mag_data['HEADING_MAG'] = np.arctan2(self.mag_data['Hy'], self.mag_data['Hx']) + np.pi/2
        self.heading_mag = self.mag_data['HEADING_MAG']
        return self.heading_mag

    @staticmethod
    def calculate_quaternions(roll, pitch, yaw):
        q0 = (np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) +
              np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))

        q1 = (np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) -
              np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))

        q2 = (np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) +
              np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2))

        q3 = (np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) -
              np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2))

        return np.array([q0, q1, q2, q3])

    def convert_to_euler(self):
        """
        Converto i roll, pitch calcolati con l'integrazione delle velocità angolari in in roll pitch e yaw
        nel sistema di riferimento corretto
        :return:
        """
        # Calcolo dei seni e coseni in modo vettoriale
        sin_roll = np.sin(self.acc_data['ROLL_KALMAN'])
        cos_roll = np.cos(self.acc_data['ROLL_KALMAN'])
        tan_pitch = np.tan(self.acc_data['PITCH_KALMAN'])
        cos_pitch = np.cos(self.acc_data['PITCH_KALMAN'])

        # Calcolo delle componenti euleriane in modo vettoriale
        roll_euler = self.gyro_data['RRATE_X'] + self.gyro_data['RRATE_Y'] * sin_roll * tan_pitch + self.gyro_data[
            'RRATE_Z'] * cos_roll * tan_pitch
        pitch_euler = self.gyro_data['RRATE_Y'] * cos_roll - self.gyro_data['RRATE_Z'] * sin_roll
        yaw_euler = self.gyro_data['RRATE_Y'] * sin_roll / cos_pitch + self.gyro_data['RRATE_Z'] * cos_roll / cos_pitch

        # Assegnazione dei risultati al DataFrame
        self.gyro_data['ROLL_EULER'] = roll_euler
        self.gyro_data['PITCH_EULER'] = pitch_euler
        self.gyro_data['YAW_EULER'] = yaw_euler

    def calculate_heading_gyro(self):
        quaternions = np.array([self.calculate_quaternions(row['ROLL_EULER'], row['PITCH_EULER'], row['YAW_EULER'])
                                for index, row in self.gyro_data.iterrows()])

        q0, q1, q2, q3 = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        self.heading_gyro = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        self.gyro_data['HEADING_GYRO'] = self.heading_gyro
        return self.heading_gyro

    def filter_heading(self):
        A = np.eye(2)
        B = np.eye(2)
        C = np.array([[1, 0]])

        Q = np.array([[0.001, 0], [0, 0.001]])
        R = 0.0001

        # Initial state with gyro heading because it's more accurate than magnetometer heading
        St = np.array([self.heading_gyro[0], 0])
        self.heading_kalman.append((self.acc_data['TIMESTAMP'].iloc[0], St[0]))

        Pt = np.eye(2)

        for t in range(1, len(self.acc_data)):
            # Predict
            St = A @ St + B @ np.array([self.heading_mag[t], 0])
            Pt = A @ Pt @ A.T + Q

            r = np.random.normal(0, np.sqrt(R))
            # Update
            Ot = self.heading_gyro[t] + r
            Kt = Pt @ C.T @ np.linalg.inv(C @ Pt @ C.T + R)
            St = St + Kt @ (Ot - C @ St)
            Pt = (np.eye(2) - Kt @ C) @ Pt
            self.heading_kalman.append((self.acc_data['TIMESTAMP'].iloc[t], St[0]))

        return self.heading_kalman


imu_processor = IMUProcessing('../data/imu_fingerprints.csv')
roll_pitch, gyro = imu_processor.load_and_process_data()
gyro.to_csv('../data/processed/imu_chain/gyro_roll_pitch_yaw.csv', index=False)

kalman_filter = KalmanFilterQuaternion(roll_pitch, gyro)
kalman_filter = pd.DataFrame(kalman_filter.apply_filter())
kalman_filter.to_csv('../data/processed/imu_chain/roll_pitch_filtered.csv', index=False)


kalman_filter = pd.read_csv('../data/processed/imu_chain/roll_pitch_filtered.csv')
print(kalman_filter.describe())
# Assuming acc_data is your accelerometer data DataFrame and it has been processed to include 'PITCH_KALMAN' and 'TIMESTAMP'
pitch_kalman_series = pd.Series(kalman_filter['PITCH_KALMAN'].values, index=kalman_filter['TIMESTAMP'])

# Instantiate the StepDetector class
step_detector = StepDetector()

# Detect steps
steps = step_detector.detect_steps(pitch_kalman_series)

# Print the number of steps detected
num_steps = len(steps)
print(f"Number of steps detected: {num_steps}")

steps_df = pd.DataFrame(steps, columns=['Max_Index', 'Min_Index'])

# Selezioniamo un intervallo di 1 minuto basato sui passi rilevati
start_index = steps_df['Max_Index'].iloc[10]
end_index = start_index + 100  # 1 minuto in campioni (10ms per campione)

# Filtriamo i dati per l'intervallo di 1 minuto
pitch_kalman_series_1min = pitch_kalman_series.iloc[start_index:end_index]
steps_dynamic_1min = steps_df[(steps_df['Max_Index'] >= start_index) & (steps_df['Min_Index'] <= end_index)]

# Visualizzazione dei risultati nell'intervallo di 1 minuto
plt.figure(figsize=(14, 7))

plt.plot(pitch_kalman_series_1min.index, pitch_kalman_series_1min, label='Pitch (Kalman)')

for _, row in steps_dynamic_1min.iterrows():
    plt.axvline(pitch_kalman_series.index[row['Max_Index']], color='r', linestyle='--', linewidth=0.5)
    plt.axvline(pitch_kalman_series.index[row['Min_Index']], color='g', linestyle='--', linewidth=0.5)

plt.xlabel('Time')
plt.ylabel('Pitch (degrees)')
plt.legend()
plt.title('Pitch with Detected Steps (Dynamic Threshold) in 1 Minute')
plt.show()

# Processing of stride length
stride_processor = StrideLengthProcessor(k=3)
imu_data = pd.read_csv('../data/imu_fingerprints.csv')

imu_processor = IMUProcessor(imu_data)
imu_data = imu_processor.filter_data()

stride_lengths = stride_processor.process_stride_length(steps, imu_data)
stride_lengths_df = pd.DataFrame(stride_lengths, columns=['TIMESTAMP', 'STRIDE_LENGTH'])
stride_lengths_df.to_csv('../data/processed/imu_chain/stride_lengths.csv', index=False)
print(stride_lengths_df.describe())

# Processing of heading
gyro = pd.read_csv('../data/processed/imu_chain/gyro_roll_pitch_yaw.csv')
heading_processor = HeadingProcessor(kalman_filter, imu_data, gyro)
kalman_heading = heading_processor.filter_heading()
kalman_heading_df = pd.DataFrame(kalman_heading, columns=['TIMESTAMP', 'HEADING'])
kalman_heading_df.to_csv('../data/processed/imu_chain/heading_filtered.csv', index=False)
#
plt.figure(figsize=(14, 7))

plt.plot(kalman_filter['TIMESTAMP'], np.degrees(heading_processor.mag_data['HEADING_MAG']), label='Heading (Magnetometer)')
plt.plot(kalman_filter['TIMESTAMP'], np.degrees(heading_processor.gyro_data['HEADING_GYRO']),
         label='Heading (Gyroscope)', linestyle='--')
plt.plot(kalman_filter['TIMESTAMP'], np.degrees(kalman_heading_df['HEADING']), label='Heading (Kalman Fused)',
         linestyle='-.')
plt.xlabel('Time')
plt.ylabel('Heading (degrees)')
plt.legend()
plt.title('Heading Estimation from Magnetometer, Gyroscope, and Kalman Filter')
plt.show()

heading = heading_processor.mag_data[['TIMESTAMP','HEADING_MAG']].copy(deep=True)

# Merge of stride lenght with imu data
merged_data = pd.merge(stride_lengths_df, heading, on='TIMESTAMP')
merged_data.to_csv('../data/processed/imu_chain/merged_data.csv', index=False)
