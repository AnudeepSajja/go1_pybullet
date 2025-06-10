import tkinter as tk
import threading

class PDGainController:
    def __init__(self):
        self.kp_values = [30.0, 35.0, 45.0]  # hip, thigh, knee
        self.kd_values = [2.5, 3.0, 5.0]
        self._start_gui_thread()

    def _start_gui_thread(self):
        thread = threading.Thread(target=self._gui_loop, daemon=True)
        thread.start()

    def _gui_loop(self):
        def make_slider(label, from_, to, resolution, initial, row, column, on_change):
            tk.Label(root, text=label).grid(row=row, column=column)
            slider = tk.Scale(root, from_=from_, to=to, resolution=resolution,
                              orient=tk.HORIZONTAL, command=on_change)
            slider.set(initial)
            slider.grid(row=row, column=column + 1)

        def on_kp1(val): self.kp_values[0] = float(val)
        def on_kp2(val): self.kp_values[1] = float(val)
        def on_kp3(val): self.kp_values[2] = float(val)
        def on_kd1(val): self.kd_values[0] = float(val)
        def on_kd2(val): self.kd_values[1] = float(val)
        def on_kd3(val): self.kd_values[2] = float(val)

        root = tk.Tk()
        root.title("6x PD Gain Controller")

        make_slider("kp1 (hip)", 0, 100, 0.1, self.kp_values[0], 0, 0, on_kp1)
        make_slider("kd1 (hip)", 0, 20, 0.1, self.kd_values[0], 0, 2, on_kd1)
        make_slider("kp2 (thigh)", 0, 100, 0.1, self.kp_values[1], 1, 0, on_kp2)
        make_slider("kd2 (thigh)", 0, 20, 0.1, self.kd_values[1], 1, 2, on_kd2)
        make_slider("kp3 (knee)", 0, 100, 0.1, self.kp_values[2], 2, 0, on_kp3)
        make_slider("kd3 (knee)", 0, 20, 0.1, self.kd_values[2], 2, 2, on_kd3)

        root.mainloop()

    def get_kp_array(self):
        return self.kp_values * 4  # repeat for 4 legs

    def get_kd_array(self):
        return self.kd_values * 4
