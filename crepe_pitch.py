import sys
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGraphicsTextItem
import librosa
import crepe

def get_equal_indexed_notes():
    NOTE_NAMES_FULL = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    indexed_notes = []
    index = 0
    for n in range(60, 89):  # MIDI 60 = C4, 88 = E6
        name = NOTE_NAMES_FULL[n % 12]
        full_name = name + str(n // 12 - 1)
        freq = 440.0 * (2 ** ((n - 69) / 12))
        indexed_notes.append((full_name, freq, index))
        index += 1
    return indexed_notes

def snap_to_note_index(freq, indexed_notes):
    return min(indexed_notes, key=lambda x: abs(freq - x[1]))

class RealTimePitchPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 실시간 음정 시각화 (자동 스크롤)")
        self.setGeometry(100, 100, 800, 400)

        self.indexed_notes = get_equal_indexed_notes()
        ticks = [(note[2], note[0]) for note in self.indexed_notes]

        self.y_axis = pg.AxisItem(orientation='left')
        self.y_axis.setTicks([ticks])
        self.plot_widget = pg.PlotWidget(axisItems={'left': self.y_axis})
        self.setCentralWidget(self.plot_widget)

        self.plot_widget.setYRange(0, len(self.indexed_notes) - 1)
        self.plot_widget.setLabel('bottom', 'Time', units='s')

        self.plot_data = self.plot_widget.plot(pen=pg.mkPen('m', width=2))

        self.note_label = QGraphicsTextItem()
        self.note_label.setDefaultTextColor(pg.mkColor('w'))
        self.note_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.plot_widget.addItem(self.note_label)
        self.note_label.setPos(5, len(self.indexed_notes) - 1)

        self.data = []
        self.elapsed_time = 0.0
        self.update_interval = 0.05
        self.x_range = 5.0
        self.current_note_name = ""

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

    def audio_callback(self, indata, frames, time, status):
        audio = indata[:, 0]
        if len(audio) < 1024:
            return

        try:
            _, freq, confidence, _ = crepe.predict(audio, self.sample_rate, viterbi=True)
            print(f"Freq: {freq[0]:.2f} Hz, Confidence: {confidence[0]:.2f}")
            if confidence[0] > 0.4:
                note_name, note_freq, note_idx = snap_to_note_index(freq[0], self.indexed_notes)
                self.data.append(note_idx)
                self.current_note_name = f"\U0001F3B5 {note_name}"
            else:
                self.data.append(np.nan)
                self.current_note_name = ""
        except Exception as e:
            print("CREPE error:", e)
            self.data.append(np.nan)
            self.current_note_name = ""

    def update_plot(self):
        self.elapsed_time += self.update_interval

        x_full = [i * self.update_interval for i in range(len(self.data))]
        start_time = max(0, self.elapsed_time - self.x_range)
        keep_start_idx = next((i for i, v in enumerate(x_full) if v >= start_time), 0)

        x = x_full[keep_start_idx:]
        y = self.data[keep_start_idx:]

        self.plot_data.setData(x, y)

        if x:
            self.plot_widget.setXRange(x[0], x[-1])
        else:
            self.plot_widget.setXRange(0, self.x_range)

        self.note_label.setPlainText(self.current_note_name)
        if x:
            print(f"🟢 x[-1]: {x[-1]:.2f}, xRange: ({x[0]:.2f} ~ {x[-1]:.2f}), y[-1]: {y[-1]}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimePitchPlot()
    window.show()
    sys.exit(app.exec_())
