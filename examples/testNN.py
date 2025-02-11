import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

# Load the dataset
file_path = '/home/anudeep/devel/workspace/src/data/go1_trot_for5mins.csv'
df = pd.read_csv(file_path)

# Split X and Y, drop unnecessary columns, and normalize
X = df.iloc[:, 1:-12]  # Exclude time column and last 12 columns for X
Y = df.iloc[:, -12:]    # Last 12 columns for Y

# Store the last four columns separately
last_four_columns = X.iloc[:, -4:]

# Normalize all columns of X except for the last four
for col in X.columns[:-4]:  # Exclude the last four columns
    col_min = X[col].min()
    col_max = X[col].max()
    
    if col_max != col_min:
        X[col] = 2 * (X[col] - col_min) / (col_max - col_min) - 1
    else:
        X[col] = 0

# Combine normalized X with the last four columns
X_normalized = pd.concat([X.iloc[:, :-4], last_four_columns], axis=1)

# Normalize all columns of Y using .loc
for col in Y.columns:
    col_min = Y[col].min()
    col_max = Y[col].max()
    
    if col_max != col_min:
        Y[col] = 2 * (Y[col] - col_min) / (col_max - col_min) - 1
    else:
        Y[col] = 0

# Drop the first two columns of X_normalized as they are not needed
input_data = X_normalized.iloc[:, 2:]  # Exclude the first two columns
output_data = Y  # Output data is already normalized

# Convert data to tensors and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(input_data.values, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(output_data.values, dtype=torch.float32).to(device)

# Split the dataset into training (70%), validation (20%), and testing (10%)
X_temp, X_test, Y_temp, Y_test = train_test_split(X_tensor, Y_tensor, test_size=0.1, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.2, random_state=42)

class NMPCPredictor(nn.Module):
    def __init__(self, input_size, output_size):
        super(NMPCPredictor, self).__init__()
        self.hidden1 = nn.Linear(input_size, 2048)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(2048, 2048)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(2048, 2048)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(2048, output_size)

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        return self.output(x)

# Initialize the model
input_size = X_tensor.shape[1]
output_size = Y_tensor.shape[1]
model = NMPCPredictor(input_size, output_size).to(device)

# Define the loss function and the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()  # L2 loss

# Training parameters
num_epochs = 500
patience = 10  
last_loss = float('inf')  
patience_counter = 0  

def evaluate_model(loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# Create DataLoaders for training and validation sets
train_loader = DataLoader(TensorDataset(X_train.to(device), Y_train.to(device)), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val.to(device), Y_val.to(device)), batch_size=256)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='logs')

# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    # Log training loss to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

    # Evaluate on validation set after each epoch
    val_loss_after_training = evaluate_model(val_loader)
    writer.add_scalar('Loss/val', val_loss_after_training, epoch)

    print(f'Epoch: {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss_after_training:.4f}')

    # Early stopping logic
    if abs(loss.item() - last_loss) < 1e-6:
        patience_counter += 1  
    else:
        patience_counter = 0  

    last_loss = loss.item()  

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1} due to no change in training loss.')
        break

# Final evaluation on the test set after training
test_loader = DataLoader(TensorDataset(X_test.to(device), Y_test.to(device)), batch_size=256)
average_test_loss = evaluate_model(test_loader)

print(f'Final Average Test Loss: {average_test_loss:.4f}')

# Close the TensorBoard writer
writer.close()
