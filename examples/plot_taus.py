import pandas as pd
import matplotlib.pyplot as plt
import os

# === File Paths ===
inference_path = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/inference_log_kickstart_online.csv"
eval_path = "/home/anudeep/devel/workspace/src/data/trot_with_vdes/with_phase/go1_trot_data_actions_eval.csv"
output_dir = "/home/anudeep/devel/workspace/data_humanoids/linference_logs/plots"
os.makedirs(output_dir, exist_ok=True)

# === Load CSVs ===
df_inf = pd.read_csv(inference_path)
df_eval = pd.read_csv(eval_path)

# === Time Vector ===
SIM_DT = 0.001
time_inf = df_inf.index * SIM_DT
time_eval = df_eval.index * SIM_DT

# === Generic Plotter ===
def plot_quantity(df, time, quantity='tau', source='inference', filename=None):
    fig, axs = plt.subplots(4, 3, figsize=(18, 10))
    axs = axs.ravel()

    for i in range(12):
        col = f"{quantity}_{i+1}" if f"{quantity}_{i+1}" in df.columns else f"a{i+1}"
        axs[i].plot(time, df[col], color='blue' if source == 'inference' else 'orange')
        axs[i].set_title(f"{quantity.upper()} Joint {i+1} ({source})")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Value")
        axs[i].grid(True)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        print(f"[✔] Saved plot: {filename}")
    plt.show()

# === Plot Inference ===
plot_quantity(df_inf, time_inf, quantity='tau', source='inference',
              filename=os.path.join(output_dir, "tau_inference.png"))
plot_quantity(df_inf, time_inf, quantity='q_des', source='inference',
              filename=os.path.join(output_dir, "q_des_inference.png"))

# === Plot Eval ===
plot_quantity(df_eval, time_eval, quantity='tau', source='eval',
              filename=os.path.join(output_dir, "tau_eval.png"))
plot_quantity(df_eval, time_eval, quantity='q_des', source='eval',
              filename=os.path.join(output_dir, "q_des_eval.png"))
