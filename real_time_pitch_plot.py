import sys
import numpy as np
import sounddevice as sd
import crepe
import pyqtgraph as pg
import parselmouth
from parselmouth.praat import call
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
import librosa

SAMPLE_RATE = 16000
BUFFER_DURATION = 1.5
BUFFER_SIZE = int(SAMPLE_RATE * BUFFER_DURATION)
VISUAL_WINDOW = 100

PITCH_MIN = librosa.note_to_hz('C3')  # 130.81 Hz
PITCH_MAX = librosa.note_to_hz('F5')  # 698.46 Hz

def cents_error(f0, f_ref):
    if f0 <= 0 or f_ref <= 0 or np.isnan(f0) or np.isnan(f_ref):
        return np.nan
    return 1200 * np.log2(f0 / f_ref)

def hz_to_note_name(hz):
    if hz <= 0 or np.isnan(hz):
        return "-"
    return librosa.hz_to_note(hz, octave=True)

def analyze_voice(buffer, sr):
    snd = parselmouth.Sound(buffer, sampling_frequency=sr)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return jitter * 100, shimmer * 100

class CREPEAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 CREPE + 발성 분석기 (C3 ~ F5)")
        self.setGeometry(100, 100, 800, 600)

        self.plot_widget = pg.PlotWidget(title="Pitch (Hz)")
        self.plot_widget.setYRange(PITCH_MIN, PITCH_MAX)
        self.curve = self.plot_widget.plot(np.zeros(VISUAL_WINDOW), pen='y')

        note_labels = [f'{n}{o}' for o in range(3, 6) for n in ['C', 'D', 'E', 'F', 'G', 'A', 'B']]
        note_labels = [n for n in note_labels if PITCH_MIN <= librosa.note_to_hz(n) <= PITCH_MAX]
        note_ticks = [librosa.note_to_hz(n) for n in note_labels]
        ticks = [(f, n) for f, n in zip(note_ticks, note_labels)]
        self.plot_widget.getAxis('left').setTicks([ticks])

        self.label = QLabel("🎧 분석 준비 중...")
        self.label.setStyleSheet("font-size: 16px; padding: 10px;")

        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
        self.pitch_history = np.zeros(VISUAL_WINDOW)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer = np.roll(self.audio_buffer, -len(indata))
        self.audio_buffer[-len(indata):] = indata[:, 0]

    def update_plot(self):
        try:
            audio = self.audio_buffer.astype(np.float32)
            _, freq_array, confidence, _ = crepe.predict(
                audio, SAMPLE_RATE, viterbi=True, step_size=100
            )
            pitch = freq_array[-1] if confidence[-1] > 0.5 and PITCH_MIN <= freq_array[-1] <= PITCH_MAX else 0.0
        except:
            pitch = 0.0

        note_name = hz_to_note_name(pitch)
        target_freq = librosa.note_to_hz(note_name) if note_name != "-" else 0
        cent = cents_error(pitch, target_freq)
        cent_text = f"{cent:+.1f} cents" if not np.isnan(cent) else "-"

        if self.timer.remainingTime() % 2000 < 150:
            try:
                jitter, shimmer = analyze_voice(self.audio_buffer, SAMPLE_RATE)
                self.label.setText(
                    f"🎵 음정: {note_name} ({pitch:.1f} Hz), 센트 오차: {cent_text}\n"
                    f"📊 Jitter: {jitter:.2f}%, Shimmer: {shimmer:.2f}%"
                )
            except Exception as e:
                self.label.setText("분석 오류: " + str(e))

        self.pitch_history = np.roll(self.pitch_history, -1)
        self.pitch_history[-1] = pitch if pitch > 0 else np.nan
        self.curve.setData(self.pitch_history)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CREPEAnalyzer()
    window.show()
    sys.exit(app.exec_())
