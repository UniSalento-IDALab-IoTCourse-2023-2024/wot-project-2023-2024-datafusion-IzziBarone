import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


def plot_accelerations(acc_original, acc_filtered, title='Accelerazione', xlabel='Sample Index',
                       ylabel='Accelerazione (m/s^2)'):
    plt.figure(figsize=(10, 6))
    plt.plot(acc_original, label='Originale', alpha=0.7)
    plt.plot(acc_filtered, label='Filtrata', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show()


class IMUProcessor:
    def __init__(self, data, sampling_rate=50):

        self.imu_data = data

        self.sampling_rate = self._calculate_sampling_rate()

        self.threshold = 0.6

        self._K = 15

        self._fs = sampling_rate
        self._cutoff = 4.0
        self._order = 3

        self.stride_lengths = []
        self.headings = []

        #self._preprocess_data()

    def _calculate_sampling_rate(self):
        # Calcola la differenza tra i timestamp consecutivi
        time_diffs = self.imu_data['TIMESTAMP'].diff().dropna()

        # Calcola l'intervallo medio di tempo in millisecondi
        mean_interval_ms = time_diffs.mean()

        # Converti l'intervallo medio di tempo in secondi
        mean_interval_s = mean_interval_ms / 1000

        # Calcola il sampling rate come inverso dell'intervallo medio di tempo in secondi
        return 1 / mean_interval_s

    def _low_pass_filter(self, data, cutoff=4, order=4):
        nyquist = 0.5 * self._fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # Check if the length of data is sufficient for filtfilt
        if len(data) <= (3 * order):
            raise ValueError(f"The length of the input vector must be greater than {3 * self._order}")

        y = filtfilt(b, a, data)
        return y

    def _project_to_ENU(self):

        # Convert angles from degrees to radians
        roll = np.deg2rad(self.imu_data['ROLL'].values)
        pitch = np.deg2rad(self.imu_data['PITCH'].values)
        yaw = np.deg2rad(self.imu_data['YAW'].values)

        acc_x_original = self.imu_data['ACC_X'].values
        acc_y_original = self.imu_data['ACC_Y'].values
        acc_z_original = self.imu_data['ACC_Z'].values

        acc_x = self._low_pass_filter(self.imu_data['ACC_X'].values, cutoff=2)
        acc_y = self._low_pass_filter(self.imu_data['ACC_Y'].values, cutoff=2)
        acc_z = self._low_pass_filter(self.imu_data['ACC_Z'].values, cutoff=2)

        plot_accelerations(acc_x_original, acc_x, title='Accelerazione X')
        plot_accelerations(acc_y_original, acc_y, title='Accelerazione Y')
        plot_accelerations(acc_z_original, acc_z, title='Accelerazione Z')

        # Initialize arrays for projected accelerations
        acc_e = np.zeros_like(acc_x)
        acc_n = np.zeros_like(acc_x)
        acc_u = np.zeros_like(acc_x)

        for i in range(len(roll)):
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll[i]), -np.sin(roll[i])],
                [0, np.sin(roll[i]), np.cos(roll[i])]
            ])

            R_y = np.array([
                [np.cos(pitch[i]), 0, np.sin(pitch[i])],
                [0, 1, 0],
                [-np.sin(pitch[i]), 0, np.cos(pitch[i])]
            ])

            R_z = np.array([
                [np.cos(yaw[i]), -np.sin(yaw[i]), 0],
                [np.sin(yaw[i]), np.cos(yaw[i]), 0],
                [0, 0, 1]
            ])

            R = np.dot(R_z, np.dot(R_y, R_x))
            acc = np.array([acc_x[i], acc_y[i], acc_z[i]])
            acc_enu = np.dot(R, acc)

            acc_e[i] = acc_enu[0]
            acc_n[i] = acc_enu[1]
            acc_u[i] = acc_enu[2]

        acc_e_original = acc_e
        acc_n_original = acc_n
        acc_u_original = acc_u

        self.imu_data['ACC_E'] = self._low_pass_filter(acc_e, cutoff=3, order=2)
        self.imu_data['ACC_N'] = self._low_pass_filter(acc_n, cutoff=3, order=2)
        self.imu_data['ACC_U'] = self._low_pass_filter(acc_u, cutoff=3, order=2)

        plot_accelerations(acc_e_original, self.imu_data['ACC_E'].values, title='Accelerazione E')
        plot_accelerations(acc_n_original, self.imu_data['ACC_N'].values, title='Accelerazione N')
        plot_accelerations(acc_u_original, self.imu_data['ACC_U'].values, title='Accelerazione U')

    def filter_data(self):
        """
            Filters the accelerometer data to project it onto the East-North-Up (ENU) coordinate system and returns the filtered IMU data.

            This method first calls the `_project_to_ENU` method to apply a low-pass filter on the accelerometer data and project it onto the ENU coordinate system.
            It then returns the filtered IMU data, which includes the projected accelerations along the East, North, and Up directions refiltered using a low-pass filter.

            Returns:
                pandas.DataFrame: The filtered IMU data with projected accelerations in the ENU coordinate system.
            """
        self._project_to_ENU()
        return self.get_imu_data()

    def get_imu_data(self):
        return self.imu_data

    def detect_steps(self):
        self._project_to_ENU()
        acc_u = self.imu_data['ACC_U'].values

        steps = []
        min_interval = 400  # Minimum interval between steps in milliseconds
        last_max = None
        last_min = None

        for i in range(1, len(acc_u) - 1):
            if acc_u[i - 1] < acc_u[i] > acc_u[i + 1]:  # local maximum
                if last_min is not None:
                    if (self.imu_data['TIMESTAMP'].iloc[i] - self.imu_data['TIMESTAMP'].iloc[
                        last_min] > min_interval) and (acc_u[i] - acc_u[last_min] > self.threshold):
                        steps.append(i)
                        last_max = i
                else:
                    last_max = i
            elif acc_u[i - 1] > acc_u[i] < acc_u[i + 1]:  # local minimum
                if last_max is not None:
                    if (self.imu_data['TIMESTAMP'].iloc[i] - self.imu_data['TIMESTAMP'].iloc[
                        last_max] > min_interval) and (acc_u[last_max] - acc_u[i] > self.threshold):
                        steps.append(i)
                        last_min = i
                else:
                    last_min = i

        return steps

    def calculate_stride_length_and_heading(self, steps=None):
        if steps is None:
            steps = self.detect_steps()

        timestamps = self.imu_data['TIMESTAMP'].values
        K = self._K

        for i in range(0, len(steps)):
            # Deve prendere i valori di acc_u tra lo step corrente e quello precedente, se lo step precedente non esiste (out of index) si parte dall'inizio del dataset
            start_idx = steps[i - 1] if i > 1 else 0
            end_idx = steps[i]
            step_acc = self.imu_data['ACC_U'].values[start_idx:end_idx]

            stride_length = K * (np.max(step_acc) - np.min(step_acc)) ** 1 / 4

            stride_length_mean = 0.846964
            q = 0.2

            if stride_length > stride_length_mean + q or stride_length < stride_length_mean - q:
                stride_length = stride_length_mean

            self.stride_lengths.append((timestamps[end_idx], stride_length))

            # Compute heading using numerical integration over the E-N plane accelerations
            heading = self._compute_heading(start_idx, end_idx)
            self.headings.append((timestamps[end_idx], heading))
            self.headings.append((timestamps[end_idx], self.imu_data['YAW'].values[end_idx]))

    def _compute_heading(self, start_idx, end_idx):
        acc_e = self.imu_data['ACC_E'].values[start_idx:end_idx]
        acc_n = self.imu_data['ACC_N'].values[start_idx:end_idx]
        time_diff = (self.imu_data['TIMESTAMP'].values[end_idx] - self.imu_data['TIMESTAMP'].values[
            start_idx]) / 1000.0  # Convert ms to seconds

        # Trapezoidal integration over the latest 1.4 seconds (or two steps)
        if time_diff > 1.4:
            integration_window = int(1.4 * self.sampling_rate)
            acc_e = acc_e[-integration_window:]
            acc_n = acc_n[-integration_window:]

        # Numerical integration using the trapezoidal rule
        vel_e = np.trapezoid(acc_e) / self.sampling_rate
        vel_n = np.trapezoid(acc_n) / self.sampling_rate

        heading = np.arctan2(vel_n, vel_e)
        return np.degrees(heading)

    def get_stride_lengths(self, dataframe=False):
        if dataframe:
            return pd.DataFrame(self.stride_lengths, columns=['TIMESTAMP', 'STRIDE_LENGTH'])
        return self.stride_lengths

    def get_headings(self, dataframe=False):
        if dataframe:
            return pd.DataFrame(self.headings, columns=['TIMESTAMP', 'HEADING'])
        return self.headings


class PositionEstimator:
    def __init__(self, initial_timestamp=0, initial_position=(0, 0)):
        self.initial_position = initial_position
        self.positions = [(initial_timestamp, initial_position[0], initial_position[1])]

    def estimate_positions(self, stride_lengths, headings):
        current_position = np.array(self.initial_position)

        for (timestamp, stride_length), (ts, heading) in zip(stride_lengths, headings):
            if timestamp != ts:
                raise ValueError("Timestamp mismatch between stride lengths and headings")
            # Convert heading from degrees to radians
            heading_rad = heading
            # Calculate new position
            new_position = current_position + np.array([
                stride_length * np.cos(heading_rad),
                stride_length * np.sin(heading_rad)
            ])
            self.positions.append((timestamp, new_position[0], new_position[1]))
            current_position = new_position

    def get_positions(self):
        return self.positions


def process():
    # Example usage - IMU PROCESSOR
    imu_data = pd.read_csv('../data/imu_fingerprints.csv')

    imu_processor = IMUProcessor(imu_data)
    imu_processor.calculate_stride_length_and_heading()
    imu_processor.get_stride_lengths(dataframe=True)
    imu_processor.get_headings(dataframe=True)
    stride_heading = (pd.merge(imu_processor.get_stride_lengths(dataframe=True), imu_processor.get_headings(dataframe=True),
                               on='TIMESTAMP')
                      .to_csv('../data/processed/imu_pos_estimate/stride_heading.csv', index=False))

    df_heading = imu_processor.get_headings(dataframe=True)
    first_timestamp = df_heading['TIMESTAMP'].iloc[0]

    stride_lengths = imu_processor.get_stride_lengths()
    print("Stride lengths:")
    print(imu_processor.get_stride_lengths(dataframe=True).describe())
    headings = imu_processor.get_headings()
    print("Headings:")
    print(imu_processor.get_headings(dataframe=True).describe())

    # Example usage - POSITION ESTIMATOR
    # Trovare il primo timestamp negli heading
    # Cercare nei dati reali la posizione corrispondente a quel timestamp
    initial_position = imu_data.loc[imu_data['TIMESTAMP'] == first_timestamp, ['X', 'Y']].iloc[0]
    position_estimator = PositionEstimator(initial_timestamp=first_timestamp,
                                           initial_position=(initial_position['X'], initial_position['Y']))
    position_estimator.estimate_positions(stride_lengths, headings)
    positions = position_estimator.get_positions()

    pd.DataFrame(positions, columns=['TIMESTAMP', 'X', 'Y']).to_csv('../data/processed/positions.csv', index=False)

    merge_data = pd.merge(pd.read_csv('../data/processed/positions.csv'), imu_data, on=['TIMESTAMP'])
    print("Done")
    merge_data.to_csv('../data/processed/merged_data.csv', index=False)
