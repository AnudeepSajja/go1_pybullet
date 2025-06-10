import pandas as pd
import numpy as np

# Load the CSV file
file_path = "/home/anudeep/devel/workspace/src/data/trot_with_vdes/go1_trot_data_actions_eval.csv"
output_path = "/home/anudeep/devel/workspace/src/data/trot_with_vdes/with_phase/go1_trot_data_actions_phase_eval.csv"

df = pd.read_csv(file_path)

# Extract foot contact states
foot_contacts = df[["foot_1", "foot_2", "foot_3", "foot_4"]].values.astype(int)

# Define trot gait cycle patterns
pattern_start = np.array([0, 1, 1, 0])
gait_phase = np.zeros(len(df))

# Find indices of all [0,1,1,0] occurrences (cycle starts)
cycle_indices = []
for i in range(1, len(foot_contacts)):
    if np.array_equal(foot_contacts[i], pattern_start):
        cycle_indices.append(i)

# Print cycle start/end index pairs
print(f"Detected {len(cycle_indices)} cycle(s):")
for i in range(len(cycle_indices) - 1):
    start = cycle_indices[i]
    end = cycle_indices[i + 1]
    if end - start > 1:
        print(f"  Cycle {i+1}: start = {start}, end = {end} → length = {end - start}")
        gait_phase[start:end] = np.linspace(0, 1, end - start, endpoint=False)
        if start > 0:
            gait_phase[start - 1] = 1.0


# Handle trailing segment after last cycle
if len(cycle_indices) >= 2:
    last_start = cycle_indices[-1]
    gait_phase[last_start:] = np.linspace(0, 1, len(df) - last_start, endpoint=False)
    if last_start > 0:
        gait_phase[last_start - 1] = 1.0

# Insert 'gait_phase' column after foot_4
insert_index = df.columns.get_loc("foot_4") + 1
df.insert(insert_index, "gait_phase", gait_phase)

# Save output
df.to_csv(output_path, index=False)
print(f"\n[✔] Gait phase saved to {output_path}")
