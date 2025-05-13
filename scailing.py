import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGraphicsTextItem
import crepe

NOTE_NAMES_FULL = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def generate_scaling_sequence():
    sequence = []
    for base_midi in range(60, 80):
        ascending = [base_midi + i for i in range(5)]
        descending = ascending[-2::-1]
        full_scale = ascending + descending
        sequence.append(full_scale)
    return sequence

def midi_to_note_name(midi):
    return NOTE_NAMES_FULL[midi % 12] + str(midi // 12 - 1)

def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12))

def snap_to_midi(freq):
    midi = 69 + 12 * np.log2(freq / 440.0)
    return int(round(midi))

class ScailingTrainer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎵 스케일링 발성 연습기")
        self.setGeometry(100, 100, 800, 400)

        self.expected_sequence = generate_scaling_sequence()
        self.current_index = 0
        self.current_scale = self.expected_sequence[self.current_index]
        self.user_sequence = []

        self.y_axis = pg.AxisItem(orientation='left')
        self.plot_widget = pg.PlotWidget(axisItems={'left': self.y_axis})
        self.setCentralWidget(self.plot_widget)
        self.plot_widget.setLabel('bottom', 'Time (s)')

        self.plot_data = self.plot_widget.plot(pen=pg.mkPen('c', width=2))
        self.guide_bars = []  # stores (bar, x_start, x_end, midi)
        self.note_label = QGraphicsTextItem()
        self.note_label.setDefaultTextColor(pg.mkColor('w'))
        self.note_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.plot_widget.addItem(self.note_label)

        self.data = []
        self.elapsed_time = 0.0
        self.update_interval = 0.05
        self.x_range = 10.0
        self.current_note_text = ""

        self.sample_rate = 16000
        self.block_size = 2048

        self.stream = sd.InputStream(
            callback=self.audio_callback,
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=self.block_size
        )
        self.stream.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(int(self.update_interval * 1000))

        self.scale_step_index = 0
        self.note_timer = QTimer()
        self.note_timer.timeout.connect(self.next_note_in_scale)

        self.set_scale_range()
        self.start_scale_timing()

    def set_scale_range(self):
        base = self.current_scale[0]
        top = base + 4
        center = base + 2
        self.plot_widget.setYRange(base - 0.5, top + 0.5)
        y_labels = [(i, midi_to_note_name(i)) for i in range(base, top + 1)]
        self.y_axis.setTicks([y_labels])
        self.note_label.setPos(5, center)
        self.plot_widget.setXRange(0, self.x_range)

    def start_scale_timing(self):
        self.scale_step_index = 0
        self.user_sequence = []
        self.elapsed_time = 0.0
        self.data = []
        for bar, *_ in self.guide_bars:
            self.plot_widget.removeItem(bar)
        self.guide_bars.clear()
        self.note_timer.start(1000)

    def next_note_in_scale(self):
        if self.scale_step_index < len(self.current_scale):
            midi = self.current_scale[self.scale_step_index]
            bar_width = 1.0  # seconds
            x_end = self.x_range
            x_start = x_end - bar_width
            bar = pg.PlotDataItem(x=[x_start, x_end], y=[midi, midi], pen=pg.mkPen('g', width=6))
            self.plot_widget.addItem(bar)
            self.guide_bars.append((bar, x_start, x_end, midi))
            self.scale_step_index += 1
            self.note_timer.start(1000)
        else:
            self.advance_scale()

    def advance_scale(self):
        self.current_index = (self.current_index + 1) % len(self.expected_sequence)
        self.current_scale = self.expected_sequence[self.current_index]
        self.set_scale_range()
        self.start_scale_timing()
        print("\n▶️ New scale:", [midi_to_note_name(m) for m in self.current_scale])

    def audio_callback(self, indata, frames, time, status):
        audio = indata[:, 0]
        if len(audio) < 1024:
            return

        try:
            _, freq, confidence, _ = crepe.predict(audio, self.sample_rate, viterbi=True)
            if confidence[0] > 0.5:
                midi = snap_to_midi(freq[0])
                self.data.append(midi)
                self.current_note_text = midi_to_note_name(midi)
                self.user_sequence.append(midi)
                self.check_pitch_match()
            else:
                self.data.append(np.nan)
                self.current_note_text = ""
        except Exception as e:
            print("CREPE error:", e)
            self.data.append(np.nan)
            self.current_note_text = ""

    def check_pitch_match(self):
        expected = self.current_scale[:len(self.user_sequence)]
        if self.user_sequence == expected:
            print("✅ Scale match!")
        elif len(self.user_sequence) <= len(expected):
            mismatch = any(abs(u - e) > 1 for u, e in zip(self.user_sequence, expected))
            if mismatch:
                print("❌ Incorrect note detected")

    def update_plot(self):
        self.elapsed_time += self.update_interval

        # shift bars to left
        updated_bars = []
        for bar, x_start, x_end, midi in self.guide_bars:
            x_start -= self.update_interval
            x_end -= self.update_interval
            if x_end < 0:
                self.plot_widget.removeItem(bar)
            else:
                bar.setData([x_start, x_end], [midi, midi])
                updated_bars.append((bar, x_start, x_end, midi))
        self.guide_bars = updated_bars

        # update user plot
        x_full = [i * self.update_interval for i in range(len(self.data))]
        keep_start_idx = next((i for i, v in enumerate(x_full) if v >= 0), 0)
        x = [self.x_range - (self.elapsed_time - i * self.update_interval) for i in range(keep_start_idx, len(self.data))]
        y = self.data[keep_start_idx:]
        self.plot_data.setData(x, y)
        self.plot_widget.setXRange(0, self.x_range)
        self.note_label.setPlainText(self.current_note_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScailingTrainer()
    window.show()
    sys.exit(app.exec_())
