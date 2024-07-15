import numpy as np
import pandas as pd
from matplotlib import image as mpimg, pyplot as plt
import ResultsPerformance as rp


class KalmanFilter:
    """
        A class to represent a Kalman Filter for tracking and predicting the state of a dynamic system.

        Attributes:
            dt (float): Time step between measurements.
            Q (numpy.ndarray): Process noise covariance matrix.
            R_pos (numpy.ndarray): Observation noise covariance matrix for position measurements.
            R_imu_acc (numpy.ndarray): Observation noise covariance matrix for IMU acceleration measurements.
            R_imu_rot (numpy.ndarray): Observation noise covariance matrix for IMU rotation measurements.
            R_imu_ang_vel (numpy.ndarray): Observation noise covariance matrix for IMU angular velocity measurements.
            x (numpy.ndarray): Initial state estimate.
            P (numpy.ndarray): Initial estimate covariance matrix.
            H_pos (numpy.ndarray): Measurement matrix for position measurements.
            H_imu_acc (numpy.ndarray): Measurement matrix for IMU acceleration measurements.
            H_imu_rot (numpy.ndarray): Measurement matrix for IMU rotation measurements.
            H_imu_ang_vel (numpy.ndarray): Measurement matrix for IMU angular velocity measurements.
            _F (numpy.ndarray): State transition matrix.

        Methods:
            predict():
                Predicts the next state of the system using the model dynamics.

            update(z, H, R):
                Updates the state estimate using a new measurement.

            apply_filter(observations_pos, observations_imu_acc, observations_imu_rot, observations_imu_ang_vel):
                Applies the Kalman filter to a series of measurements.

            rotation_matrix(roll, pitch, yaw):
                Calculates the rotation matrix from roll, pitch, and yaw angles.
        """

    def __init__(self, init_state):
        """
        Initializes the Kalman Filter with the given time step and initial state.
        :param dt: Time step between measurements.
        :param initial_state: Initial state estimate.
        """
        self.Q = np.eye(3) * 100  # Covarianza del rumore di processo
        self.R_pos = np.eye(2) * 10000  # Covarianza del rumore di osservazione per la posizione stimata
        self.x = init_state
        self.P = np.eye(3) * 0.01

        self.H_pos = np.array([[1, 0, 0],
                               [0, 1, 0]])

        self._F = np.eye(3)

    def predict(self, step_length, heading):
        """
            Predicts the next state of the system using the state transition matrix (F).

            This method updates the state estimate (x) and the estimate covariance (P)
            based on the model's dynamics. The process noise covariance (Q) is also added
            to the estimate covariance to account for the uncertainty in the process.
        """
        # Aggiornare lo stato con la lunghezza del passo e l'heading
        delta_x = step_length * np.cos(np.deg2rad(heading))
        delta_y = step_length * np.sin(np.deg2rad(heading))

        noise_delta_x = np.random.normal(loc=0, scale=0.001)
        noise_delta_y = np.random.normal(loc=0, scale=0.001)
        noise_heading = np.random.normal(loc=0, scale=0.1)

        # Matrice di transizione dello stato
        self._F = np.array([[1, 0, -delta_y],
                            [0, 1, delta_x],
                            [0, 0, 1]])

        # Predizione dello stato
        self.x[0] += delta_x + noise_delta_x
        self.x[1] += delta_y + noise_delta_y
        self.x[2] = heading + noise_heading

        self.P = np.dot(np.dot(self._F, self.P), self._F.T) + self.Q

    def update(self, z, H, R):
        """
            Updates the state estimate using a new measurement (z).

            :param z: The new measurement vector.
            :param H: The measurement matrix, which maps the state estimate into the measurement space.
            :param R: The measurement noise covariance matrix for the current measurement.

            This method calculates the Kalman gain (K), updates the state estimate (x),
            and updates the estimate covariance (P) to reflect the new measurement.
            """
        y = z.reshape(-1, 1) - np.dot(H, self.x)
        S = np.dot(np.dot(H, self.P), H.T) + R
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        return y

    def apply_filter(self, observations_pos, imu_data, window_size=1000):
        """
            Applies the Kalman filter to a series of measurements.

            :param observations_pos: Array of position measurements.
            :param imu_data: Array of IMU measurements (acceleration, rotation, angular velocity).
            :param window_size: Size of the time window (ms) for processing IMU data.

            :return: A numpy array of estimated positions after applying the Kalman filter to the measurements.

            This method sequentially processes each measurement, updating the state estimate
            and estimate covariance with each new measurement. It handles different types of
            measurements (position, IMU acceleration, IMU rotation, IMU angular velocity) by
            using the appropriate measurement matrix (H) and noise covariance (R) for each.
        """
        estimated_positions = []
        innovations = []

        for i in range(len(observations_pos)):

            # Finestra temporale scorrevole per i dati IMU
            current_time = observations_pos[i, 0]
            window_start = current_time - window_size / 2
            window_end = current_time + window_size / 2

            window_imu_data = imu_data[(imu_data[:, 0] >= window_start) & (imu_data[:, 0] <= window_end)]

            for imu_observation in window_imu_data:
                stride_length = imu_observation[1]
                heading = imu_observation[2]
                self.predict(stride_length, heading)


            # Aggiornamento con i dati di posizione WiFi/BLE
            innovation = self.update(observations_pos[i, 1:], self.H_pos, self.R_pos)
            innovations.append(innovation.flatten())

            estimated_positions.append((current_time, self.x[:2, 0][0], self.x[:2, 0][1]))

        return np.array(estimated_positions)

    @staticmethod
    def rotation_matrix(roll, pitch, yaw):
        """
        Calculates the rotation matrix from roll, pitch, and yaw angles.
        :param roll:
        :param pitch:
        :param yaw:
        :return: Rotation matrix
        """
        R_matrix = np.array([
            [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)],
            [np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
             np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
             np.sin(roll) * np.cos(pitch)],
            [np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw),
             np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw),
             np.cos(roll) * np.cos(pitch)]
        ])
        return R_matrix


obs_pos = pd.read_csv("../data/positions/merged_positions.csv")
# Convert TIMESTAMP datetime column in milliseconds
obs_pos['TIMESTAMP'] = obs_pos['TIMESTAMP'].astype(int)
obs_pos = obs_pos.sort_values(by='TIMESTAMP')

imu_data = pd.read_csv("../data/processed/imu_pos_estimate/stride_heading.csv")
imu_data['TIMESTAMP'] = imu_data['TIMESTAMP'].astype(int)
imu_data = imu_data.sort_values(by='TIMESTAMP')

# Initialize the Kalman Filter
initial_state = np.zeros((3, 1))
N = 5
# Calcolo della media mobile delle prime N osservazioni per inizializzare lo stato
initial_pos = obs_pos[['X', 'Y']].iloc[:N].mean().values
initial_state[0] = initial_pos[0]
initial_state[1] = initial_pos[1]

position_data_np = obs_pos[['TIMESTAMP', 'x', 'y']].to_numpy()
imu_data_np = imu_data[
    ['TIMESTAMP', 'STRIDE_LENGTH', 'HEADING']].to_numpy()

kalman_filter = KalmanFilter(init_state=initial_state)
estimated_positions = kalman_filter.apply_filter(position_data_np, imu_data_np, window_size=500)

e_df = pd.DataFrame(estimated_positions, columns=['TIMESTAMP', 'x', 'y'])
e_df.sort_values(by='TIMESTAMP')
e_df_renamed = e_df.rename(columns={'x': 'KF_x', 'y': 'KF_y'})
# Esegue il merge di e_df_renamed con obs_pos su TIMESTAMP
kf_merged = pd.merge(e_df_renamed, obs_pos, on='TIMESTAMP')
kf_merged.to_csv('../data/positions/KF_estimated_positions.csv',index=False)
error_metrics = rp.ErrorMetrics(kf_merged,estimated_cols=['KF_x', 'KF_y'], true_cols=['X', 'Y'])
error_metrics.print_errors()


print("Estimated positions:")
print(e_df[['x', 'y']].describe())
print("\nOriginal positions:")
print(obs_pos[['X', 'Y']].describe())


# Plot the estimated positions
def plot_results(merged_data, estimated_positions):
    # Carica l'immagine della planimetria
    img = mpimg.imread('../data/datapoints.png')

    x_min, x_max = -3, 50
    y_min, y_max = -1, 38
    extent = (x_min, x_max, y_min, y_max)

    # Plot delle posizioni originali e stimate
    original_positions = merged_data[['X', 'Y']].values
    estimated_positions = estimated_positions[['x', 'y']].values

    plt.figure(figsize=(10, 6))
    # Mostra l'immagine della planimetria
    plt.imshow(img, extent=extent, aspect='auto')
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], marker='o', alpha=0.5, color='purple',
             label='Estimated Positions (Kalman)')
    plt.plot(original_positions[:, 0], original_positions[:, 1], label='Original Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Confronto tra Posizioni Originali e Stimate (Kalman)')
    plt.legend()
    plt.grid(True)
    plt.show()


#plot_results(obs_pos, obs_pos)
plot_results(obs_pos, e_df)