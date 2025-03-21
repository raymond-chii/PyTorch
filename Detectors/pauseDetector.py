import numpy as np
import librosa
import matplotlib.pyplot as plt

class PauseDetector:

    def __init__(self, threshold=0.01, min_pause_duration=0.2):
        self.threshold = threshold
        self.min_pause_duration = min_pause_duration


    def detect_pauses(self, file):

        y, sr = librosa.load(file)

        hop_length = int(sr*0.01)
        frame_rate = int(sr/hop_length)
        rms = librosa.feature.rms(y=y, frame_length=frame_rate, hop_length=hop_length)[0]

        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        
        pauses = rms < self.threshold
        
        pause_segments = []
        in_pause = False
        start_pause = 0
        
        for i, is_pause in enumerate(pauses):
            if is_pause and not in_pause:
                in_pause = True
                start_pause = times[i]
            elif not is_pause and in_pause:
                in_pause = False
                duration = times[i] - start_pause
                if duration >= self.min_pause_duration:
                    pause_segments.append((start_pause, times[i]))
        
        if in_pause:
            duration = times[-1] - start_pause
            if duration >= self.min_pause_duration:
                pause_segments.append((start_pause, times[-1]))
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, rms)
        plt.axhline(y=self.threshold, color='r', linestyle='--', label='Threshold')
        
        for start, end in pause_segments:
            plt.axvspan(start, end, color='red', alpha=0.3)
            
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.title('Pause Detection')
        plt.legend()
        plt.show()
        
        return pause_segments

detector = PauseDetector(threshold=0.015, min_pause_duration=0.2)
pauses = detector.detect_pauses("PROCESS-V1/Process-rec-001/Process-rec-001__CTD.wav")
print(f"Detected {len(pauses)} pauses:")
for i, (start, end) in enumerate(pauses):
    print(f"Pause {i+1}: {start:.2f}s - {end:.2f}s (duration: {end-start:.2f}s)")