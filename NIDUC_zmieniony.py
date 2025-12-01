import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

from matplotlib.figure import Figure


class SmoothingVoter:
    def __init__(self, tolerance=10.0, alpha=0.1):
        self.tolerance = tolerance
        self.alpha = alpha
        self.previous_output = None

    def process(self, measurements):
        filtered_measurements = [m for m in measurements if m is not None]
        if not filtered_measurements:
            return 0.0  # brak ważnych pomiarów, zwróć 0 lub inną wartość domyślną

        med = np.median(filtered_measurements)
        valid_measurements = [m for m in filtered_measurements if abs(m - med) <= self.tolerance]
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


class MOutOfNVoter:
    def __init__(self, default_tolerance=10.0, default_max_deviation=5.0):
        self.default_tolerance = default_tolerance
        self.default_max_deviation = default_max_deviation

    def vote(self, values, M, tolerance=None, true_pressure=None, max_deviation=None):
        if tolerance is None:
            tolerance = self.default_tolerance
        if max_deviation is None:
            max_deviation = self.default_max_deviation

        filtered = [(i, v) for i, v in enumerate(values) if v is not None]

        for i, val_i in filtered:
            count = 1
            group_indices = [i]
            for j, val_j in filtered:
                if j != i and abs(val_i - val_j) <= tolerance:  # POPRAWIONE !=
                    count += 1
                    group_indices.append(j)

            if count >= M:
                group_values = [values[idx] for idx in group_indices]
                avg = sum(group_values) / len(group_values)

                if true_pressure is None or abs(avg - true_pressure) <= max_deviation:
                    excluded = [idx for idx in range(len(values)) if idx not in group_indices]
                    return avg, group_indices, excluded
                else:
                    return None, [], list(range(len(values)))

        return None, [], list(range(len(values)))


class SensorSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulacja Voterów - Porównanie Smoothing vs M-out-of-N")

        # Utwórz obie klasy voterów
        self.smoothing_voter = SmoothingVoter(tolerance=15.0, alpha=0.15)
        self.m_out_of_n_voter = MOutOfNVoter(default_tolerance=15.0, default_max_deviation=5.0)

        self.max_len = 100
        self.data_s1 = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_s2 = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_s3 = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_true = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_smoothing = deque([0] * self.max_len, maxlen=self.max_len)
        self.data_m_out_of_n = deque([0] * self.max_len, maxlen=self.max_len)

        # Panel sterowania
        control_panel = ttk.Frame(root, padding="10")
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # Rzeczywiste ciśnienie
        ttk.Label(control_panel, text="RZECZYWISTE CIŚNIENIE",
                  font=('Arial', 10, 'bold')).pack(pady=(0, 10))
        self.true_pressure_var = tk.DoubleVar(value=50.0)
        pres_frame = ttk.Frame(control_panel)
        pres_frame.pack(fill=tk.X)
        self.lbl_pres_val = ttk.Label(pres_frame, text="50.0 Bar", width=8, anchor="e")
        self.lbl_pres_val.pack(side=tk.RIGHT)
        scale_pres = ttk.Scale(pres_frame, from_=0, to=100, variable=self.true_pressure_var,
                               orient='horizontal')
        scale_pres.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.true_pressure_var.trace_add("write", lambda *args:
        self.lbl_pres_val.config(text=f"{self.true_pressure_var.get():.1f} Bar"))

        ttk.Separator(control_panel, orient='horizontal').pack(fill=tk.X, pady=20)

        # Sensory
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

            def update_offset_label(*args, v=offset_var, l=lbl_offset_val):
                l.config(text=f"{v.get():+.1f}", foreground="red" if abs(v.get()) > 0.1 else "gray")

            offset_var.trace_add("write", update_offset_label)

            broken_var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(frame, text="AWARIA (Brak sygnału)", variable=broken_var)
            chk.pack(anchor='w')

            self.sensors_controls.append({"offset": offset_var, "broken": broken_var})

        ttk.Button(control_panel, text="Resetuj Błędy (Wszystko na 0)",
                   command=self.reset_errors).pack(pady=20, fill=tk.X)

        # Parametry M-out-of-N
        self.M_var = tk.IntVar(value=2)

        self.status_label = ttk.Label(control_panel, text="Status: ---", font=('Arial', 9))
        self.status_label.pack(pady=5)

        # Wykres
        # Wykres - DWA SUBPLOTY OBOK SIEBIE
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(10, 5))
        self.ax_smoothing = self.fig.add_subplot(121)  # 1 wiersz, 2 kolumny, 1. wykres
        self.ax_m_out_of_n = self.fig.add_subplot(122)  # 1 wiersz, 2 kolumny, 2. wykres

        # === PIERWSZY WYKRES - SMOOTHING VOTER ===
        self.line_true_s = self.ax_smoothing.plot([], [], 'k:', label='Rzeczywiste', alpha=0.5, linewidth=2)[0]
        self.line_s1_s = self.ax_smoothing.plot([], [], 'r--', label='Sensor 1', alpha=0.8)[0]
        self.line_s2_s = self.ax_smoothing.plot([], [], 'g--', label='Sensor 2', alpha=0.8)[0]
        self.line_s3_s = self.ax_smoothing.plot([], [], 'b--', label='Sensor 3', alpha=0.8)[0]
        self.line_smoothing = self.ax_smoothing.plot([], [], 'm-', linewidth=3, label='Smoothing Voter')[0]

        self.ax_smoothing.set_title("Smoothing Voter", fontweight='bold')
        self.ax_smoothing.set_ylim(-10, 110)
        self.ax_smoothing.set_xlim(0, self.max_len)
        self.ax_smoothing.set_ylabel("Ciśnienie [Bar]")
        self.ax_smoothing.legend(loc='upper left')
        self.ax_smoothing.grid(True, alpha=0.3)

        # === DRUGI WYKRES - M-OUT-OF-N VOTER ===
        self.line_true_m = self.ax_m_out_of_n.plot([], [], 'k:', label='Rzeczywiste', alpha=0.5, linewidth=2)[0]
        self.line_s1_m = self.ax_m_out_of_n.plot([], [], 'r--', label='Sensor 1', alpha=0.8)[0]
        self.line_s2_m = self.ax_m_out_of_n.plot([], [], 'g--', label='Sensor 2', alpha=0.8)[0]
        self.line_s3_m = self.ax_m_out_of_n.plot([], [], 'b--', label='Sensor 3', alpha=0.8)[0]
        self.line_m_out_of_n = self.ax_m_out_of_n.plot([], [], 'orange', linewidth=3, label='M-out-of-N Voter')[0]

        self.ax_m_out_of_n.set_title("M-out-of-N Voter", fontweight='bold')
        self.ax_m_out_of_n.set_ylim(-10, 110)
        self.ax_m_out_of_n.set_xlim(0, self.max_len)
        self.ax_m_out_of_n.set_ylabel("Ciśnienie [Bar]")
        self.ax_m_out_of_n.legend(loc='upper left')
        self.ax_m_out_of_n.grid(True, alpha=0.3)

        self.fig.tight_layout()  # Dopasuj odstępy

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
            return None
        offset = ctrl['offset'].get()
        noise = np.random.normal(0, 1.0)
        return true_val + offset + noise

    def update_plot(self):
        true_val = self.true_pressure_var.get()
        readings = [self.get_sensor_value(i, true_val) for i in range(3)]

        smoothing_result = self.smoothing_voter.process(readings)

        m_result, accepted, excluded = self.m_out_of_n_voter.vote(
            values=readings,
            M=self.M_var.get(),
            tolerance=15.0,
            true_pressure=true_val,
            max_deviation=5.0
        )
        m_final_result = m_result if m_result is not None else np.median([v for v in readings if v is not None]) or 0.0

        self.status_label.config(
            text=f"M={self.M_var.get()}: {accepted} OK, {len(excluded)} OUT" if m_result else f"Brak grupy M={self.M_var.get()}")

        self.data_true.append(true_val)
        self.data_s1.append(readings[0] if readings[0] is not None else np.nan)
        self.data_s2.append(readings[1] if readings[1] is not None else np.nan)
        self.data_s3.append(readings[2] if readings[2] is not None else np.nan)
        self.data_smoothing.append(smoothing_result)
        self.data_m_out_of_n.append(m_final_result)

        x_data = list(range(len(self.data_smoothing)))

        # Aktualizacja obrazu wykresu smoothing voter
        self.line_true_s.set_data(x_data, list(self.data_true))
        self.line_s1_s.set_data(x_data, list(self.data_s1))
        self.line_s2_s.set_data(x_data, list(self.data_s2))
        self.line_s3_s.set_data(x_data, list(self.data_s3))
        self.line_smoothing.set_data(x_data, list(self.data_smoothing))
        self.ax_smoothing.relim()
        self.ax_smoothing.autoscale_view()

        # Aktualizacja obrazu wykresu M-out-of-N
        self.line_true_m.set_data(x_data, list(self.data_true))
        self.line_s1_m.set_data(x_data, list(self.data_s1))
        self.line_s2_m.set_data(x_data, list(self.data_s2))
        self.line_s3_m.set_data(x_data, list(self.data_s3))
        self.line_m_out_of_n.set_data(x_data, list(self.data_m_out_of_n))
        self.ax_m_out_of_n.relim()
        self.ax_m_out_of_n.autoscale_view()

        self.canvas.draw()

        self.root.after(50, self.update_plot)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x700")
    app = SensorSimulationApp(root)
    root.mainloop()
