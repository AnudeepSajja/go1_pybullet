import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load CSV Log ===
log_path = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_logs_online.csv"
df = pd.read_csv(log_path)

# === Time Vector ===
SIM_DT = 0.001  # 1 kHz
time = df.index * SIM_DT

# === Plot Predicted Joint Positions ===
def plot_predicted_joints(df, time, filename=None):
    fig, axs = plt.subplots(4, 3, figsize=(16, 10))
    axs = axs.ravel()

    for i in range(12):
        col = f"q_des_{i+1}"
        axs[i].plot(time, df[col])
        axs[i].set_title(f"Predicted Joint {i+1}")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Position [rad]")
        axs[i].grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[✔] Saved plot: {filename}")
    # plt.show()
    plt.close(fig)  # <- add this line

# === Plot Torques ===
def plot_torques(df, time, filename=None):
    fig, axs = plt.subplots(4, 3, figsize=(16, 10))
    axs = axs.ravel()

    for i in range(12):
        col = f"tau_{i+1}"
        axs[i].plot(time, df[col])
        axs[i].set_title(f"Torque {i+1}")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Torque [Nm]")
        axs[i].grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[✔] Saved plot: {filename}")
    plt.show()

# === Plot Gait Phase ===
def plot_phase(df, time, filename=None):
    if 'phase' in df.columns:
        plt.figure(figsize=(10, 3))
        plt.plot(time, df['phase'], color='purple')
        plt.title("Gait Phase Over Time")
        plt.xlabel("Time [s]")
        plt.ylabel("Phase [0–1]")
        plt.grid(True)
        plt.tight_layout()
        if filename:
            plt.savefig(filename)
            print(f"[✔] Saved plot: {filename}")
        # plt.show()
        plt.close()  # <- add this line
    else:
        print("[!] 'phase' column not found.")

# === Output Directory ===
output_dir = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/plots"
os.makedirs(output_dir, exist_ok=True)

# === Generate Plots ===
plot_predicted_joints(df, time, os.path.join(output_dir, "predicted_joint_positions.png"))
plot_torques(df, time, os.path.join(output_dir, "joint_torques.png"))
plot_phase(df, time, os.path.join(output_dir, "gait_phase_plot.png"))
