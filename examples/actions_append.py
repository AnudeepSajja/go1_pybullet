import csv
import os

# Define the KP and KV constants
kp_hip = 75.0
kv_hip = 5.0

kp_thigh = 75.0
kv_thigh = 5.0

kp_knee = 75.0
kv_knee = 5.0

# Directory containing the 20 CSV files
# input_file = "/home/anudeep/devel/workspace/src/data/jump_data/go1_trot_jump_data_20.csv"
# input_file = "/home/anudeep/go1_mpc/data_files/go1_lowlevel_data_20.csv"
input_file = "/home/anudeep/devel/workspace/src/data/trot_data_pd_policy_2901/go1_trot_data_actions_eval.csv"
# output_file_path = "/home/anudeep/devel/workspace/src/data/jump_data_actions/"
# output_file_path = "/home/anudeep/go1_mpc/lowlevel_actions/2
output_file_path = "/home/anudeep/devel/workspace/src/data/trot_data_pd_policy_2901/actions_append/"

# Ensure the output directory exists
os.makedirs(output_file_path, exist_ok=True)

csv_file = output_file_path + 'go1_trot_data_actions_eval.csv'

# open the output file for writing
with open(csv_file, mode='w', newline='') as out_file:
    writer = csv.writer(out_file)
    
    # Define the header for the output file
    header = ["time", "base_pos_x", "base_pos_y", "base_pos_z", "base_ori_x", "base_ori_y", "base_ori_z", "base_ori_w",
              "base_vel_x", "base_vel_y", "base_vel_z", 
              "imu_acc_x", "imu_acc_y", "imu_acc_z",
              "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
              "qj_1", "qj_2", "qj_3", "qj_4", "qj_5", "qj_6", "qj_7", "qj_8", "qj_9", "qj_10", "qj_11", "qj_12",
              "dqj_1", "dqj_2", "dqj_3", "dqj_4", "dqj_5", "dqj_6", "dqj_7", "dqj_8", "dqj_9", "dqj_10", "dqj_11", "dqj_12",
              "foot_1", "foot_2", "foot_3", "foot_4",
              "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6", "tau_7", "tau_8", "tau_9", "tau_10", "tau_11", "tau_12",
              "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"]
    
    # Write the header to the output file
    writer.writerow(header)

    with open(input_file, mode='r') as in_file:
        reader = csv.reader(in_file)
        # Skip the header of the current CSV file
        next(reader)
        
        for row in reader:
            # Extract the data from the current row
            time = row[0]
            base_pos = row[1:4]  # base_pos_x, base_pos_y, base_pos_z
            base_ori = row[4:8]  # base_ori_x, base_ori_y, base_ori_z, base_ori_w
            base_vel = row[8:11]  # base_vel_x, base_vel_y, base_vel_z
            imu_acc = row[11:14]  # imu_acc_x, imu_acc_y, imu_acc_z
            imu_gyro = row[14:17]  # imu_gyro_x, imu_gyro_y, imu_gyro_z
            qj = row[17:29]  # qj_1 to qj_12
            dqj = row[29:41]  # dqj_1 to dqj_12
            foot_contact = row[41:45]  # foot_1 to foot_4
            tau = row[45:57]  # tau_1 to tau_12
            
            # Calculate a1 to a12
            a = []
            # for j in range(12):
            #     qj_j = float(qj[j])
            #     dqj_j = float(dqj[j])
            #     tau_j = float(tau[j])
            #     a_j = qj_j + (tau_j + kv * dqj_j) / kp
            #     a.append(a_j)
            
            for j in range(12):
                qj_j = float(qj[j])
                dqj_j = float(dqj[j])
                tau_j = float(tau[j])
                if j in [0, 3, 6, 9]:  # hip joints
                    a_j = qj_j + (tau_j + kv_hip * dqj_j) / kp_hip
                elif j in [1, 2, 7, 10]:  # thigh joints
                    a_j = qj_j + (tau_j + kv_thigh * dqj_j) / kp_thigh
                else:  # knee joints
                    a_j = qj_j + (tau_j + kv_knee * dqj_j) / kp_knee
                a.append(a_j)

            # Create the new row with a1 to a12 appended
            new_row = row + a  # Append a1 to a12 to the original row
            
            # Write the new row to the output file
            writer.writerow(new_row)

print(f"Data saved successfully to {csv_file}")

 