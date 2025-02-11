import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == "__main__":
    # Read the Data
    path = "/home/khorshidi/go1_workspace/workspace/test_data/"
    motion_name = "bound_test"
    cent_state = np.loadtxt(path+motion_name+"_cent_state.dat", delimiter='\t', dtype=np.float32)
    input_vector = np.loadtxt(path+motion_name+"_input_vector.dat", delimiter='\t', dtype=np.float32)
    
    tx = np.arange(input_vector.shape[1])
    plt.plot(tx, input_vector[14, :], "b", linewidth=1.0)
    plt.grid()
    plt.show()