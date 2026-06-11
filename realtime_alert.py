import os
import queue
import librosa
import numpy as np
import cv2
import sounddevice as sd
import tensorflow as tf
from collections import deque
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import hashlib

# =====================================================================
# OPTIMIZED CONFIGURATION
# =====================================================================
@dataclass
class Config:
    model_path: str = 'environmental_noise_cnn.h5'
    sample_rate: int = 22050
    duration: float = 2.0
    window_size: int = int(22050 * 2)
    
    # Model dimensions
    input_height: int = 40
    input_width: int = 174
    
    # Detection parameters
    min_confidence: float = 0.60
    smoothing_window: int = 5
    detection_hold: float = 0.8
    
    # Performance
    target_fps: int = 30
    audio_blocksize: int = 1024
    max_queue_size: int = 20
    
    # Display
    window_width: int = 800
    window_height: int = 500

# UrbanSound8K Classes
CLASSES = [
    "Air Conditioner", "Car Horn", "Children Playing", 
    "Dog Bark", "Drilling", "Engine Idling", 
    "Gun Shot", "Jackhammer", "Siren", "Street Music"
]

HAZARDOUS_SOUNDS = {"Car Horn", "Siren", "Gun Shot", "Dog Bark"}

# Global queue
audio_queue = queue.Queue(maxsize=Config.max_queue_size)

# =====================================================================
# FIXED CACHED PREPROCESSOR
# =====================================================================
class CachedPreprocessor:
    """Cache preprocessing results to avoid redundant computation"""
    def __init__(self, max_cache_size=3):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.cache_keys = deque(maxlen=max_cache_size)
    
    def get_audio_hash(self, audio_buffer):
        """Create a simple hash of the audio buffer for caching"""
        # Take a subset of samples for hashing (faster)
        step = max(1, len(audio_buffer) // 100)
        samples = audio_buffer[::step].flatten()
        
        # Convert to bytes for hashing
        sample_bytes = samples.tobytes()
        return hashlib.md5(sample_bytes).hexdigest()
    
    def preprocess(self, audio_buffer, preprocess_func):
        """Preprocess with caching"""
        audio_hash = self.get_audio_hash(audio_buffer)
        
        if audio_hash in self.cache:
            return self.cache[audio_hash]
        
        result = preprocess_func(audio_buffer)
        
        # Manage cache size
        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.cache_keys[0]
            del self.cache[oldest_key]
        
        self.cache[audio_hash] = result
        self.cache_keys.append(audio_hash)
        
        return result

# =====================================================================
# OPTIMIZED AUDIO PREPROCESSING
# =====================================================================
def preprocess_audio_fast(audio_buffer, max_pad_len=174):
    """Optimized preprocessing with fewer operations"""
    audio_mono = np.squeeze(audio_buffer)
    
    # Quick normalization
    max_val = np.abs(audio_mono).max()
    if max_val > 1e-6:
        audio_mono = audio_mono / max_val
    
    # Generate Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=audio_mono, 
        sr=Config.sample_rate, 
        n_mels=40,
        n_fft=2048,
        hop_length=512,
        center=False
    )
    
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Fast pad/crop
    current_len = spectrogram_db.shape[1]
    if current_len < max_pad_len:
        pad_width = max_pad_len - current_len
        spectrogram_db = np.pad(spectrogram_db, ((0, 0), (0, pad_width)), mode='constant')
    elif current_len > max_pad_len:
        spectrogram_db = spectrogram_db[:, :max_pad_len]
        
    return spectrogram_db.reshape(1, 40, max_pad_len, 1)

# =====================================================================
# PREDICTION SMOOTHER
# =====================================================================
class PredictionSmoother:
    """Efficient prediction smoothing"""
    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
        
    def update(self, predictions):
        """Update with moving average"""
        self.history.append(predictions)
        
        if len(self.history) == 0:
            return predictions
        
        return np.mean(self.history, axis=0)

# =====================================================================
# DETECTION MANAGER
# =====================================================================
class DetectionManager:
    """Manages detection state with persistence"""
    def __init__(self, min_confidence=0.60, hold_time=0.8):
        self.min_confidence = min_confidence
        self.hold_time = hold_time
        
        self.current_detection = None
        self.current_confidence = 0.0
        self.last_detection_time = 0.0
        
    def update(self, detected_sound: str, confidence: float, current_time: float) -> Tuple[Optional[str], float]:
        """Update detection state"""
        
        # New detection with sufficient confidence
        if confidence >= self.min_confidence:
            self.current_detection = detected_sound
            self.current_confidence = confidence
            self.last_detection_time = current_time
        
        # Check if we should still show the detection
        elif current_time - self.last_detection_time > self.hold_time:
            self.current_detection = None
            self.current_confidence = 0.0
        
        return self.current_detection, self.current_confidence
    
    def get_top_predictions(self, predictions: np.ndarray, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N predictions efficiently"""
        indices = np.argsort(predictions)[::-1][:n]
        return [(CLASSES[i], predictions[i]) for i in indices]

# =====================================================================
# OPTIMIZED RENDERER
# =====================================================================
class OptimizedRenderer:
    """Efficient rendering"""
    def __init__(self, width=800, height=500):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
    def render(self, 
               detection: Optional[str], 
               confidence: float, 
               top_predictions: List[Tuple[str, float]],
               fps: int, 
               queue_size: int, 
               history_size: int,
               threshold: float) -> np.ndarray:
        """Render the complete frame"""
        
        # Clear canvas
        self.canvas.fill(0)
        
        # Draw title bar
        cv2.rectangle(self.canvas, (0, 0), (self.width, 50), (30, 30, 30), -1)
        cv2.putText(self.canvas, "DeafAssist AI - Real-time Sound Detection", (50, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(self.canvas, f"FPS:{fps}", (700, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Draw main detection area
        if detection and confidence >= threshold:
            is_hazardous = detection in HAZARDOUS_SOUNDS
            
            if is_hazardous:
                # Hazard alert with pulsing effect
                pulse = int((np.sin(time.time() * 12) + 1) / 2 * 60)
                self.canvas[:180, :] = (0, 0, 80 + pulse)
                cv2.rectangle(self.canvas, (10, 55), (self.width-10, 185), (0, 0, 255), 4)
                cv2.putText(self.canvas, "⚠️ HAZARD DETECTED ⚠️", (250, 95), 
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(self.canvas, detection.upper(), (250, 160), 
                           cv2.FONT_HERSHEY_TRIPLEX, 1.6, (0, 255, 255), 3)
            else:
                # Normal detection
                self.canvas[:180, :] = (0, 50, 0)
                cv2.rectangle(self.canvas, (50, 55), (self.width-50, 185), (0, 255, 0), 3)
                cv2.putText(self.canvas, "✓ SOUND DETECTED", (300, 95), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(self.canvas, detection, (250, 160), 
                           cv2.FONT_HERSHEY_TRIPLEX, 1.4, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_width = int(450 * confidence)
            cv2.rectangle(self.canvas, (180, 200), (180 + bar_width, 225), (0, 255, 0), -1)
            cv2.rectangle(self.canvas, (180, 200), (630, 225), (100, 100, 100), 2)
            cv2.putText(self.canvas, f"{confidence*100:.0f}%", (645, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Monitoring mode
            self.canvas[:150, :] = (20, 20, 20)
            cv2.putText(self.canvas, "🎤 Monitoring Environment", (270, 120), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (100, 100, 100), 2)
            cv2.putText(self.canvas, f"Threshold: {threshold*100:.0f}%", (300, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        
        # Draw top predictions
        y_start = 260
        cv2.putText(self.canvas, "Top Predictions:", (40, y_start), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        for i, (sound, prob) in enumerate(top_predictions[:5]):
            y = y_start + 40 + i * 38
            
            # Draw probability bar
            bar_width = int(500 * prob)
            color = (0, 255, 0) if prob >= threshold else (80, 80, 80)
            cv2.rectangle(self.canvas, (50, y), (50 + bar_width, y + 22), color, -1)
            cv2.rectangle(self.canvas, (50, y), (550, y + 22), (60, 60, 60), 1)
            
            # Draw text
            text_color = (255, 255, 255) if prob >= threshold else (150, 150, 150)
            cv2.putText(self.canvas, f"{i+1}. {sound[:18]}", (60, y + 16), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            cv2.putText(self.canvas, f"{prob*100:.0f}%", (560, y + 16), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            # Draw indicator if above threshold
            if prob >= threshold:
                cv2.putText(self.canvas, "✓", (600, y + 16), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Draw status bar
        cv2.rectangle(self.canvas, (0, self.height-35), (self.width, self.height), (30, 30, 30), -1)
        status = f"Queue:{queue_size} | History:{history_size} | Threshold:{threshold*100:.0f}% | [+]/[-] Adjust"
        cv2.putText(self.canvas, status, (15, self.height-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        controls = "[Q]uit | [R]eset"
        cv2.putText(self.canvas, controls, (self.width-150, self.height-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        
        return self.canvas

# =====================================================================
# PERFORMANCE STATS
# =====================================================================
class PerformanceStats:
    """Track performance metrics"""
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.prediction_times = deque(maxlen=30)
        
    def add_frame_time(self, dt: float):
        self.frame_times.append(dt)
    
    def add_prediction_time(self, dt: float):
        self.prediction_times.append(dt)
    
    @property
    def avg_fps(self) -> float:
        if not self.frame_times:
            return 0
        return 1.0 / (np.mean(self.frame_times) + 0.001)
    
    @property
    def avg_prediction_ms(self) -> float:
        if not self.prediction_times:
            return 0
        return np.mean(self.prediction_times) * 1000

# =====================================================================
# AUDIO CALLBACK
# =====================================================================
def audio_callback(indata, frames, time, status):
    """Optimized callback with error handling"""
    if status:
        return
    
    try:
        if not audio_queue.full():
            audio_queue.put(indata.copy())
    except:
        pass

# =====================================================================
# MAIN APPLICATION
# =====================================================================
class SoundDetectionApp:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.smoother = PredictionSmoother(window_size=self.config.smoothing_window)
        self.detection_manager = DetectionManager(
            min_confidence=self.config.min_confidence,
            hold_time=self.config.detection_hold
        )
        self.renderer = OptimizedRenderer(self.config.window_width, self.config.window_height)
        self.preprocessor = CachedPreprocessor()
        self.stats = PerformanceStats()
        
        self.audio_buffer = np.zeros((self.config.window_size, 1), dtype=np.float32)
        self.last_frame_time = time.time()
        self.running = True
        
    def load_model(self):
        """Load the TensorFlow model"""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        
        print("Loading model...", end=" ", flush=True)
        self.model = tf.keras.models.load_model(self.config.model_path)
        print("✓")
        
    def run(self):
        """Main application loop"""
        print("\n" + "="*60)
        print("DeafAssist AI - Optimized Sound Detection System")
        print("="*60)
        print(f"Threshold: {self.config.min_confidence*100:.0f}% (Press +/- to adjust)")
        print(f"Smoothing: {self.config.smoothing_window} frames")
        print(f"Detection hold: {self.config.detection_hold}s")
        print("="*60 + "\n")
        
        # Start audio stream
        stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            callback=audio_callback,
            blocksize=self.config.audio_blocksize,
            latency='low'
        )
        stream.start()
        
        print("🎤 Microphone active! Ready to detect sounds...\n")
        
        # Main loop
        frame_duration = 1.0 / self.config.target_fps
        last_prediction_time = 0
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Process audio chunks
                chunks_processed = 0
                while not audio_queue.empty() and chunks_processed < 10:
                    chunk = audio_queue.get()
                    self.audio_buffer = np.roll(self.audio_buffer, -len(chunk), axis=0)
                    self.audio_buffer[-len(chunk):] = chunk
                    chunks_processed += 1
                
                # Only predict at target FPS
                if frame_start - last_prediction_time >= frame_duration:
                    last_prediction_time = frame_start
                    
                    # Preprocess (with caching)
                    pred_start = time.time()
                    input_tensor = self.preprocessor.preprocess(self.audio_buffer, preprocess_audio_fast)
                    
                    # Predict
                    predictions = self.model.predict(input_tensor, verbose=0)[0]
                    self.stats.add_prediction_time(time.time() - pred_start)
                    
                    # Smooth predictions
                    smoothed = self.smoother.update(predictions)
                    
                    # Get detection
                    class_id = np.argmax(smoothed)
                    confidence = smoothed[class_id]
                    detected_sound = CLASSES[class_id]
                    
                    # Update detection manager
                    current_time = time.time()
                    display_detection, display_confidence = self.detection_manager.update(
                        detected_sound, confidence, current_time
                    )
                    
                    # Get top predictions
                    top_preds = self.detection_manager.get_top_predictions(smoothed, 5)
                    
                    # Calculate FPS
                    frame_time = time.time() - frame_start
                    self.stats.add_frame_time(frame_time)
                    
                    # Render and display
                    canvas = self.renderer.render(
                        detection=display_detection,
                        confidence=display_confidence,
                        top_predictions=top_preds,
                        fps=int(self.stats.avg_fps),
                        queue_size=audio_queue.qsize(),
                        history_size=len(self.smoother.history),
                        threshold=self.config.min_confidence
                    )
                    
                    cv2.imshow("DeafAssist AI", canvas)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('r'):
                    self.smoother.history.clear()
                    print("✓ Reset prediction history")
                elif key == ord('+') or key == ord('='):
                    self.config.min_confidence = min(0.95, self.config.min_confidence + 0.05)
                    self.detection_manager.min_confidence = self.config.min_confidence
                    print(f"Threshold: {self.config.min_confidence*100:.0f}%")
                elif key == ord('-'):
                    self.config.min_confidence = max(0.30, self.config.min_confidence - 0.05)
                    self.detection_manager.min_confidence = self.config.min_confidence
                    print(f"Threshold: {self.config.min_confidence*100:.0f}%")
                
                # Small sleep to prevent CPU spike
                elapsed = time.time() - frame_start
                if elapsed < frame_duration:
                    time.sleep(max(0, frame_duration - elapsed - 0.005))
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            stream.stop()
            stream.close()
            cv2.destroyAllWindows()
            print("\n✓ System shutdown complete")

# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    app = SoundDetectionApp()
    app.load_model()
    app.run()