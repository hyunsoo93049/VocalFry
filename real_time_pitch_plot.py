import sys
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

try:
    import sounddevice as sd
except OSError as e:
    print("[오류] sounddevice 모듈에서 PortAudio 라이브러리를 찾을 수 없습니다.")
    print("macOS에서는 'brew install portaudio' 후 'pip install sounddevice'를 실행하세요.")
    sys.exit(1)


# 🎯 C4 ~ E6 (자연음만) + 등간격 인덱스 기반
def get_equal_indexed_notes():
    NOTE_NAMES_FULL = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    WHOLE_NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    indexed_notes = []
    index = 0
    for n in range(60, 89):  # MIDI 60 = C4, 88 = E6
        name = NOTE_NAMES_FULL[n % 12]
        if name in WHOLE_NOTES:
            full_name = name + str(n // 12 - 1)
            freq = 440.0 * (2 ** ((n - 69) / 12))
            indexed_notes.append((full_name, freq, index))
            index += 1
    return indexed_notes



def snap_to_note_index(freq, indexed_notes):
    return min(indexed_notes, key=lambda x: abs(freq - x[1]))  # returns (name, freq, index)


def detect_pitch(audio_data, sample_rate):
    windowed = audio_data * np.hanning(len(audio_data))
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1 / sample_rate)
    magnitude = np.abs(fft)
    peak_idx = np.argmax(magnitude)
    return freqs[peak_idx]


class RealTimePitchPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 실시간 보컬 음정 그래프 (등간격)")
        self.setGeometry(100, 100, 800, 400)

        # 🎼 등간격 음계 생성
        self.indexed_notes = get_equal_indexed_notes()

        # 🎯 y축 라벨: index -> note_name
        ticks = [(note[2], note[0]) for note in self.indexed_notes]
        self.y_axis = pg.AxisItem(orientation='left')
        self.y_axis.setTicks([ticks])
        self.plot_widget = pg.PlotWidget(axisItems={'left': self.y_axis})
        self.setCentralWidget(self.plot_widget)

        # 🎯 y축 index 범위 고정
        self.plot_widget.setYRange(0, len(self.indexed_notes) - 1)

        self.plot_data = self.plot_widget.plot(pen='m')
        self.data = []
        self.max_length = 100

        self.sample_rate = 44100
        self.block_size = 1024

        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size
        )
        self.stream.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

    def audio_callback(self, indata, frames, time, status):
        audio = indata[:, 0]
        freq = detect_pitch(audio, self.sample_rate)

        # 🎯 주파수가 범위 내일 때만
        if 196.0 <= freq <= 1318.51:
            note_name, note_freq, note_idx = snap_to_note_index(freq, self.indexed_notes)
            self.data.append(note_idx)
        else:
            self.data.append(np.nan)

        if len(self.data) > self.max_length:
            self.data.pop(0)

    def update_plot(self):
        self.plot_data.setData(self.data)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealTimePitchPlot()
    window.show()
    sys.exit(app.exec_())
