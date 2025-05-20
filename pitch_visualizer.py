import sys
import numpy as np
import sounddevice as sd
import librosa
import pyqtgraph as pg
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer

# 설정
SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_LENGTH = 512
BUFFER_SIZE = 100

def hz_to_note_name(hz):
    if hz <= 0 or np.isnan(hz):
        return "-"
    return librosa.hz_to_note(hz, octave=True)

class RealTimePitchPlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 실시간 음정 시각화")
        self.setGeometry(100, 100, 800, 400)

        # 🎯 플롯 설정
        self.plot_widget = pg.PlotWidget(title="Pitch: -")
        self.setCentralWidget(self.plot_widget)

        # 🎯 y축 범위: C4~B4에 해당하는 주파수
        self.plot_widget.setYRange(librosa.note_to_hz('C4'), librosa.note_to_hz('B4'))

        # 🎯 y축에 도~시 표시
        note_labels = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']
        note_ticks = [librosa.note_to_hz(n) for n in note_labels]
        ticks = [(freq, note) for freq, note in zip(note_ticks, note_labels)]
        self.plot_widget.getAxis('left').setTicks([ticks])

        # 초기 그래프 데이터
        self.data = np.zeros(BUFFER_SIZE)
        self.curve = self.plot_widget.plot(self.data, pen='y')

        # 오디오 버퍼
        self.audio_buffer = np.zeros(FRAME_SIZE, dtype=np.float32)

        # 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)  # 20 fps

        # 🎤 마이크 입력 스트림
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=HOP_LENGTH,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer = np.concatenate((self.audio_buffer[len(indata):], indata[:, 0]))

    def update_plot(self):
        try:
            f0, voiced_flag, _ = librosa.pyin(
                self.audio_buffer,
                fmin=librosa.note_to_hz('C4'),
                fmax=librosa.note_to_hz('B4'),
                sr=SAMPLE_RATE,
                frame_length=FRAME_SIZE,
                hop_length=HOP_LENGTH
            )
            valid_indices = (~np.isnan(f0)) & (voiced_flag)
            valid_pitch = f0[valid_indices]
            pitch = valid_pitch[-1] if valid_pitch.size > 0 else 0.0
        except Exception as e:
            print("[Error]", e)
            pitch = 0.0

        # 계이름 표시
        note_name = hz_to_note_name(pitch)
        if pitch > 0:
            self.plot_widget.setTitle(f"🎯 현재 음정: {note_name} ({pitch:.1f} Hz)")
        else:
            self.plot_widget.setTitle("Pitch: -")

        # 그래프 데이터 업데이트
        self.data = np.roll(self.data, -1)
        self.data[-1] = pitch if pitch > 0 else np.nan
        self.curve.setData(self.data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RealTimePitchPlot()
    window.show()
    sys.exit(app.exec_())
