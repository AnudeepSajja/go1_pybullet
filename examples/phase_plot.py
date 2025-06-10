# import pandas as pd
# import matplotlib.pyplot as plt

# # Load your dataset
# df = pd.read_csv("/home/anudeep/devel/workspace/data_humanoids/phase_distri/go1_trot_dataset_.csv")

# # Select only the first trajectory (first 5000 steps)
# df_traj1 = df.iloc[:10000]

# # Extract the phase and contact columns
# phase = df_traj1["phase"].values
# contacts = [df_traj1[f"contact_leg_{i}"].values for i in range(4)]
# labels = ["FL", "FR", "RL", "RR"]

# # Plot phase and contact states
# plt.figure(figsize=(12, 6))
# plt.plot(phase, label="Phase", color="black", linewidth=2)

# for i, contact in enumerate(contacts):
#     plt.plot(contact * (i + 1), label=f"{labels[i]} Contact", linestyle='--')

# plt.title("Phase and Foot Contacts - Trajectory 1")
# plt.xlabel("Timestep")
# plt.ylabel("Phase / Contact (offset)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()




#########################################################################################################


import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("/home/anudeep/devel/workspace/src/data/trot_with_vdes/go1_trot_data_actions_.csv")

# Select first trajectory (e.g., first 10,000 timesteps)
df_traj1 = df.iloc[:5000]

# Extract phase and foot contact signals
phase = df_traj1["phase"].values
contacts = [df_traj1[f"foot_{i+1}"].values for i in range(4)]
labels = ["FL", "FR", "RL", "RR"]
colors = ["blue", "orange", "green", "red"]

# Create the plot
plt.figure(figsize=(12, 6))

# Plot phase variable
plt.plot(phase, label="Phase", color="black", linestyle='--')

# # Plot all foot contacts (binary 0 or 1) in different colors
# for i, contact in enumerate(contacts):
#     plt.plot(contact, label=f"{labels[i]} Contact", color=colors[i])

# Plot only FL and RL foot contacts (binary 0 or 1) in different colors
for i, label in enumerate(["FL", "RL"]):
    plt.plot(contacts[i], label=f"{label} Contact", color=colors[i])

# Styling
plt.title("Phase and Foot Contacts (Binary)")
plt.xlabel("Timestep")
plt.ylabel("Binary Value (0 or 1)")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
