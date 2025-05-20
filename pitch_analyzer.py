import sys
import numpy as np
import sounddevice as sd
import librosa
import pyqtgraph as pg
import parselmouth
from parselmouth.praat import call
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer

# 설정값
SAMPLE_RATE = 22050
FRAME_SIZE = 2048
HOP_LENGTH = 512
BUFFER_SIZE = 100
TARGET_NOTE = 'G4'
TARGET_FREQ = librosa.note_to_hz(TARGET_NOTE)

# 센트 오차 계산
def cents_error(f0, f_ref):
    if f0 <= 0 or np.isnan(f0):
        return np.nan
    return 1200 * np.log2(f0 / f_ref)

# 음계 이름 변환
def hz_to_note_name(hz):
    if hz <= 0 or np.isnan(hz):
        return "-"
    return librosa.hz_to_note(hz, octave=True)

# Jitter / Shimmer 분석 함수
def analyze_voice(buffer, sr):
    snd = parselmouth.Sound(buffer, sampling_frequency=sr)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return jitter * 100, shimmer * 100  # 백분율로 반환

class RealTimeAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 실시간 발성 분석기")
        self.setGeometry(100, 100, 800, 600)

        # 그래프
        self.plot_widget = pg.PlotWidget(title="Pitch (Hz)")
        self.plot_widget.setYRange(librosa.note_to_hz('C4'), librosa.note_to_hz('B4'))
        self.curve = self.plot_widget.plot(np.zeros(BUFFER_SIZE), pen='y')

        # y축 눈금 (도~시)
        note_labels = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4']
        note_ticks = [librosa.note_to_hz(n) for n in note_labels]
        ticks = [(f, n) for f, n in zip(note_ticks, note_labels)]
        self.plot_widget.getAxis('left').setTicks([ticks])

        # 피드백 라벨
        self.label = QLabel("🎯 음정, 센트 오차, 발성 분석 결과가 여기에 표시됩니다.")
        self.label.setStyleSheet("font-size: 16px; padding: 10px;")

        # 레이아웃
        layout = QVBoxLayout()
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.label)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.data = np.zeros(BUFFER_SIZE)
        self.audio_buffer = np.zeros(3 * SAMPLE_RATE, dtype=np.float32)  # 3초간 누적 분석용

        # 타이머
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(50)

        # 마이크 입력
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
        self.audio_buffer = np.roll(self.audio_buffer, -len(indata))
        self.audio_buffer[-len(indata):] = indata[:, 0]

    def update_plot(self):
        try:
            f0, voiced_flag, _ = librosa.pyin(
                self.audio_buffer[-FRAME_SIZE*4:],  # 최근 0.4초
                fmin=librosa.note_to_hz('C4'),
                fmax=librosa.note_to_hz('B4'),
                sr=SAMPLE_RATE,
                frame_length=FRAME_SIZE,
                hop_length=HOP_LENGTH
            )
            valid = (~np.isnan(f0)) & voiced_flag
            pitch = f0[valid][-1] if np.any(valid) else 0.0
        except:
            pitch = 0.0

        note_name = hz_to_note_name(pitch)
        cent = cents_error(pitch, TARGET_FREQ)
        cent_text = f"{cent:+.1f} cents" if not np.isnan(cent) else "-"

        # 2초 주기로 발성 분석
        if self.timer.remainingTime() % 2000 < 60:
            try:
                jitter, shimmer = analyze_voice(self.audio_buffer, SAMPLE_RATE)
                self.label.setText(
                    f"🎵 현재 음정: {note_name} ({pitch:.1f} Hz), 센트 오차: {cent_text}\n"
                    f"📊 Jitter: {jitter:.2f}%, Shimmer: {shimmer:.2f}%"
                )
            except Exception as e:
                self.label.setText("분석 오류: " + str(e))

        # 그래프 업데이트
        self.data = np.roll(self.data, -1)
        self.data[-1] = pitch if pitch > 0 else np.nan
        self.curve.setData(self.data)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RealTimeAnalyzer()
    window.show()
    sys.exit(app.exec_())
