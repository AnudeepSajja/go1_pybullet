import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the minimum and maximum values for v_des
v_x_min, v_x_max = -0.8, 0.8  # Replace with your actual min and max values for v_x
v_y_min, v_y_max = -0.5, 0.5   # Replace with your actual min and max values for v_y

# Calculate the mean and standard deviation for Gaussian distribution
mean_v_x = (v_x_min + v_x_max) / 2
mean_v_y = (v_y_min + v_y_max) / 2

std_dev_v_x = (v_x_max - v_x_min) / 6  # Assuming ±3 standard deviations cover the range
std_dev_v_y = (v_y_max - v_y_min) / 6

# Generate points for plotting
v_x = np.linspace(v_x_min, v_x_max, 100)
v_y = np.linspace(v_y_min, v_y_max, 100)

# Calculate the probability density functions (PDFs)
v_x_pdf = (1 / (std_dev_v_x * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v_x - mean_v_x) / std_dev_v_x) ** 2)
v_y_pdf = (1 / (std_dev_v_y * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v_y - mean_v_y) / std_dev_v_y) ** 2)

# Generate random values for v_x and v_y within their ranges
v_x_random = np.round(np.random.uniform(v_x_min, v_x_max, 20), 2)
v_y_random = np.round(np.random.uniform(v_y_min, v_y_max, 20), 2)

# Sort samples in ascending order
v_x_random.sort()
v_y_random.sort()

# Create a DataFrame and save to CSV using pandas
df = pd.DataFrame({'v_x': v_x_random, 'v_y': v_y_random, 'v_z': 0.0})
csv_file = '/home/anudeep/devel/workspace/src/data/v_des/v_des_data.csv'
df.to_csv(csv_file, index=False)

print(f"Rounded values saved to {csv_file}")

# Plotting the Gaussian distributions and random samples on a single figure
plt.figure(figsize=(10, 6))

# Plot for v_x
plt.plot(v_x, v_x_pdf, color='blue', label='PDF of $v_x$', linewidth=2)
# Plot for v_y on the same axes
plt.plot(v_y, v_y_pdf, color='orange', label='PDF of $v_y$', linewidth=2)

# Plotting the random samples from v_x
plt.scatter(v_x_random,
            (1 / (std_dev_v_x * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v_x_random - mean_v_x) / std_dev_v_x) ** 2), 
            color='red', label='Random Samples from $v_x$', zorder=5)

# Plotting the random samples from v_y
plt.scatter(v_y_random,
            (1 / (std_dev_v_y * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((v_y_random - mean_v_y) / std_dev_v_y) ** 2), 
            color='green', label='Random Samples from $v_y$', zorder=5)

# Adding titles and labels to the plot
plt.title('Gaussian Distributions of $v_x$ and $v_y$ with Random Samples')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.grid()
plt.legend()

# Save the plot as a PNG file
plot_path = '/home/anudeep/devel/workspace/src/data/v_des/v_des_plot.png'
plt.savefig(plot_path)

# Show the plot with samples
plt.tight_layout()
plt.show()
