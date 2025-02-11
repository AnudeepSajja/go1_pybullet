import numpy as np
import matplotlib.pyplot as plt

# Define your data
tau_data = np.linspace(0, 4.5, 100)  # Actual tau data from 0 to 4.5
q_cur = np.linspace(0, 1, 100)  # q_cur data from 0 to 1
v_cur = np.linspace(0, 1, 100)  # v_cur data from 0 to 1

# Define your constants
kp = 1.0
kd = 0.1

# Calculate q_des
q_des = (tau_data + kd * v_cur) / kp + q_cur

# Calculate cal_tau using the calculated q_des with different q_cur and v_cur
q_cur_new = np.linspace(0, 0.5, 100)  # q_cur data from 0 to 0.5
v_cur_new = np.linspace(0, 0.5, 100)  # v_cur data from 0 to 0.5

cal_tau_new = kp * (q_des - q_cur_new) + kd * (0 - v_cur_new)

# Plotting
plt.figure(figsize=(12, 8))

# Plot actual tau vs calculated tau
plt.subplot(2, 1, 1)
plt.plot(tau_data, label='Actual tau')
plt.plot(cal_tau_new, label='Calculated tau')
plt.title('Actual tau vs Calculated tau')

plt.show()


# # Calculate cal_tau using the calculated q_des
# cal_tau = kp * (q_des - q_cur) + kd * (0 - v_cur)

# # Plotting
# plt.figure(figsize=(12, 8))

# # Plot actual tau vs calculated tau
# plt.subplot(2, 1, 1)
# plt.plot(tau_data, label='Actual tau')
# plt.plot(cal_tau, label='Calculated tau')
# plt.title('Actual tau vs Calculated tau')
# plt.xlabel('Sample')
# plt.ylabel('Tau')
# plt.legend()
# plt.grid(True)

# # Plot the difference between actual and calculated tau
# plt.subplot(2, 1, 2)
# diff_tau = tau_data - cal_tau
# plt.plot(diff_tau)
# plt.title('Difference between Actual and Calculated tau')
# plt.xlabel('Sample')
# plt.ylabel('Difference')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# # Print some statistics
# print(f"Max difference: {np.max(np.abs(diff_tau))}")
# print(f"Mean difference: {np.mean(diff_tau)}")
# print(f"Standard deviation of difference: {np.std(diff_tau)}")
