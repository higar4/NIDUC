import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque


class SmoothingVoter:
    def __init__(self, tolerance=10.0, alpha=0.1):
        self.tolerance = tolerance
        self.alpha = alpha
        self.previous_output = None

    def process(self, measurements):
        med = np.median(measurements)
        valid_measurements = [m for m in measurements if abs(m - med) <= self.tolerance]

        if not valid_measurements:
            voted_value = med
        else:
            voted_value = np.mean(valid_measurements)

        if self.previous_output is None:
            output = voted_value
        else:
            output = (self.alpha * voted_value) + ((1 - self.alpha) * self.previous_output)

        self.previous_output = output
        return output


class SensorSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulacja Smoothing Voter - Panel Sterowania")

        self.voter = SmoothingVoter(tolerance=15.0, alpha=0.15)

        self.max_len = 100
        self.data_s1 = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_s2 = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_s3 = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_out = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_true = deque([0] * self.max_len, maxlen=self.max_len)

        control_panel = ttk.Frame(root, padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Label(control_panel, text="RZECZYWISTE CIŚNIENIE", font=('Arial', 10, 'bold')).pack(pady=(0, 10))

        self.true_pressure_var = tk.DoubleVar(value=50.0)

        pres_frame = ttk.Frame(control_panel)
        pres_frame.pack(fill=tk.X)

        self.lbl_pres_val = ttk.Label(pres_frame, text="50.0 Bar", width=8, anchor="e")
        self.lbl_pres_val.pack(side=tk.RIGHT)

        scale_pres = ttk.Scale(pres_frame, from_=0, to=100, variable=self.true_pressure_var, orient='horizontal')
        scale_pres.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.true_pressure_var.trace_add("write", lambda *args: self.lbl_pres_val.config(
            text=f"{self.true_pressure_var.get():.1f} Bar"))

        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=20)

        self.sensors_controls = []
        colors = ['red', 'green', 'blue']

        for i, col in zip(range(1, 4), colors):
            frame = ttk.LabelFrame(control_panel, text=f"Sensor {i} ({col})", padding=5)
            frame.pack(fill=tk.X, pady=5)

            offset_var = tk.DoubleVar(value=0.0)

            header_frame = ttk.Frame(frame)
            header_frame.pack(fill=tk.X)
            ttk.Label(header_frame, text="Błąd (Offset):").pack(side=tk.LEFT)
            lbl_offset_val = ttk.Label(header_frame, text="0.0", width=6, anchor="e", foreground="gray")
            lbl_offset_val.pack(side=tk.RIGHT)

            scale_offset = ttk.Scale(frame, from_=-50, to=50, variable=offset_var, orient='horizontal')
            scale_offset.pack(fill=tk.X, pady=(0, 5))

            offset_var.trace_add("write", lambda *args, v=offset_var, l=lbl_offset_val: l.config(
                text=f"{v.get():+.1f}",
                foreground="red" if abs(v.get()) > 0.1 else "gray"
            ))

            broken_var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(frame, text="AWARIA (Brak sygnału)", variable=broken_var)
            chk.pack(anchor='w')

            self.sensors_controls.append({
                "offset": offset_var,
                "broken": broken_var
            })

        ttk.Button(control_panel, text="Resetuj Błędy (Wszystko na 0)", command=self.reset_errors).pack(pady=20,
                                                                                                        fill=tk.X)

        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.line_true, = self.ax.plot([], [], 'k:', label='Rzeczywiste', alpha=0.5)
        self.line_s1, = self.ax.plot([], [], 'r--', label='Sensor 1')
        self.line_s2, = self.ax.plot([], [], 'g--', label='Sensor 2')
        self.line_s3, = self.ax.plot([], [], 'b--', label='Sensor 3')
        self.line_out, = self.ax.plot([], [], 'k-', linewidth=3, label='WYNIK (Voter)')

        self.ax.set_ylim(-10, 110)
        self.ax.set_xlim(0, self.max_len)
        self.ax.legend(loc='upper left')
        self.ax.grid(True)
        self.ax.set_title("Monitor Ciśnienia - Smoothing Voter")
        self.ax.set_ylabel("Ciśnienie [Bar]")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.update_plot()

    def reset_errors(self):
        for ctrl in self.sensors_controls:
            ctrl['offset'].set(0.0)
            ctrl['broken'].set(False)

    def get_sensor_value(self, index, true_val):
        ctrl = self.sensors_controls[index]

        if ctrl['broken'].get():
            return 0.0

        offset = ctrl['offset'].get()
        noise = np.random.normal(0, 1.0)
        return true_val + offset + noise

    def update_plot(self):
        true_val = self.true_pressure_var.get()

        readings = []
        for i in range(3):
            val = self.get_sensor_value(i, true_val)
            readings.append(val)

        result = self.voter.process(readings)

        self.data_true.append(true_val)
        self.data_s1.append(readings[0])
        self.data_s2.append(readings[1])
        self.data_s3.append(readings[2])
        self.data_out.append(result)

        x_data = range(len(self.data_out))

        self.line_true.set_data(x_data, self.data_true)
        self.line_s1.set_data(x_data, self.data_s1)
        self.line_s2.set_data(x_data, self.data_s2)
        self.line_s3.set_data(x_data, self.data_s3)
        self.line_out.set_data(x_data, self.data_out)

        self.canvas.draw()
        self.root.after(50, self.update_plot)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x650")
    app = SensorSimulationApp(root)
    root.mainloop()