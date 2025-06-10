import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load the CSV log file ===
log_path = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_log_vdes_with_phase.csv"
df = pd.read_csv(log_path)

# === Plot Joint Data Function ===
def plot_joint_data(df, kind='predicted', filename=None):
    fig, axs = plt.subplots(4, 3, figsize=(16, 10))
    axs = axs.ravel()

    for i in range(12):
        if kind == 'predicted':
            col = f'pred_joint_{i+1}'
            title = f'Predicted Joint {i+1}'
        elif kind == 'tau':
            col = f'tau_{i+1}'
            title = f'Torque {i+1}'
        elif kind == 'current':
            col = f'curr_joint_{i+1}'
            title = f'Current Joint {i+1}'
        else:
            raise ValueError("Invalid kind. Use 'predicted', 'tau', or 'current'.")

        axs[i].plot(df[col])
        axs[i].set_title(title)
        axs[i].set_xlabel("Timestep")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[✔] Saved plot: {filename}")
    plt.show()

# === Plot IMU Function ===
def plot_imu_data(df, filename=None):
    fig, axs = plt.subplots(2, 3, figsize=(14, 6))
    axs = axs.ravel()

    imu_cols = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z',
                'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z']

    for i, col in enumerate(imu_cols):
        axs[i].plot(df[col])
        axs[i].set_title(col)
        axs[i].set_xlabel("Timestep")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[✔] Saved plot: {filename}")
    plt.show()

# === Gait Phase Plot ===
def plot_gait_phase(df, filename=None):
    plt.figure(figsize=(10, 3))
    plt.plot(df['gait_phase'], label='Gait Phase', color='purple')
    plt.title('Gait Phase over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[✔] Saved plot: {filename}")
    plt.show()

# === Output Directory ===
output_dir = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/plots"
os.makedirs(output_dir, exist_ok=True)

# === Call Plotters ===
plot_joint_data(df, kind='predicted', filename=os.path.join(output_dir, "predicted_joint_positions.png"))
plot_joint_data(df, kind='tau', filename=os.path.join(output_dir, "joint_torques.png"))
plot_joint_data(df, kind='current', filename=os.path.join(output_dir, "current_joint_positions.png"))
plot_gait_phase(df, filename=os.path.join(output_dir, "gait_phase_plot.png"))
plot_imu_data(df, filename=os.path.join(output_dir, "imu_data_plot.png"))
