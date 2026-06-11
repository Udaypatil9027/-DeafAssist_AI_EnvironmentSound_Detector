import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import socket
import json
import threading
import queue
import time
import os
import sys
from collections import deque
from scipy import signal

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_PATH = 'environmental_noise_cnn.h5'
SAMPLE_RATE = 22050
DURATION = 2
WINDOW_SIZE = int(SAMPLE_RATE * DURATION)

CLASSES = [
    "Air Conditioner", "Car Horn", "Children Playing", 
    "Dog Bark", "Drilling", "Engine Idling", 
    "Gun Shot", "Jackhammer", "Siren", "Street Music"
]

HOST = '127.0.0.1'
PORT = 9999

print("="*50)
print("DeafAssist AI - High Accuracy Backend")
print("="*50)

# =====================================================================
# YOUR ORIGINAL WORKING PREPROCESSING (KEEPING EXACTLY AS IS)
# =====================================================================
def preprocess_audio_original(audio_buffer, max_pad_len=174):
    """
    YOUR ORIGINAL WORKING PREPROCESSING - DO NOT CHANGE!
    This gave you 84% confidence for Dog Bark
    """
    audio_mono = np.squeeze(audio_buffer)
    
    # Generate Mel-Spectrogram (YOUR ORIGINAL METHOD)
    spectrogram = librosa.feature.melspectrogram(y=audio_mono, sr=SAMPLE_RATE, n_mels=40)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Pad or crop to exact dimensions
    if spectrogram_db.shape[1] < max_pad_len:
        pad_width = max_pad_len - spectrogram_db.shape[1]
        spectrogram_db = np.pad(spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        spectrogram_db = spectrogram_db[:, :max_pad_len]
        
    return spectrogram_db.reshape(1, 40, max_pad_len, 1)

# =====================================================================
# ENHANCED AUDIO PREPROCESSING (WITHOUT CHANGING CORE LOGIC)
# =====================================================================
class AccuratePreprocessor:
    def __init__(self):
        # Using YOUR original preprocessing
        self.preprocess_func = preprocess_audio_original
        
    def preprocess(self, audio_buffer):
        """Uses your original working preprocessing"""
        return self.preprocess_func(audio_buffer)

# =====================================================================
# HIGH-ACCURACY AUDIO PROCESSOR
# =====================================================================
class AccurateAudioProcessor:
    def __init__(self):
        self.model = None
        self.preprocessor = AccuratePreprocessor()
        self.audio_queue = queue.Queue(maxsize=30)
        self.audio_buffer = np.zeros((WINDOW_SIZE, 1), dtype=np.float32)
        
        # Larger smoothing window for stability (like your original)
        self.prediction_history = deque(maxlen=8)
        
        # Current state
        self.current_prediction = "None"
        self.current_confidence = 0.0
        self.all_predictions = [0.0] * len(CLASSES)
        self.is_running = True
        
        # For audio enhancement
        self.noise_floor = 0.01
        
    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model not found: {MODEL_PATH}")
            return False
        
        try:
            print("Loading model...", end=" ", flush=True)
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✓")
            print(f"Model input shape: {self.model.input_shape}")
            return True
        except Exception as e:
            print(f"✗ {e}")
            return False
    
    def enhance_audio(self, audio):
        """Light enhancement without changing characteristics"""
        audio = np.squeeze(audio)
        
        # Simple DC offset removal
        audio = audio - np.mean(audio)
        
        # Gentle high-pass filter (remove only subsonic rumble)
        if len(audio) > 100:
            b, a = signal.butter(2, 20, btype='high', fs=SAMPLE_RATE)
            audio = signal.filtfilt(b, a, audio)
        
        return audio.reshape(-1, 1)
    
    def audio_callback(self, indata, frames, time, status):
        """Audio callback"""
        if status:
            return
        if not self.audio_queue.full():
            self.audio_queue.put(indata.copy())
    
    def process_stream(self):
        """Main processing loop with YOUR original method"""
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=1024,
                latency='low'
            )
            stream.start()
            print("🎤 Microphone active - Using HIGH ACCURACY mode")
            print("✓ Your original preprocessing is being used\n")
        except Exception as e:
            print(f"Microphone error: {e}")
            return
        
        last_print_time = 0
        
        while self.is_running:
            # Process audio chunks
            chunks_processed = 0
            while not self.audio_queue.empty() and chunks_processed < 10:
                chunk = self.audio_queue.get()
                chunk_len = len(chunk)
                self.audio_buffer = np.roll(self.audio_buffer, -chunk_len, axis=0)
                self.audio_buffer[-chunk_len:] = chunk
                chunks_processed += 1
            
            # YOUR ORIGINAL PREPROCESSING - UNCHANGED
            input_tensor = self.preprocessor.preprocess(self.audio_buffer)
            
            # Predict using your model
            predictions = self.model.predict(input_tensor, verbose=0)[0]
            
            # Smooth predictions (like your original)
            self.prediction_history.append(predictions)
            smoothed = np.mean(self.prediction_history, axis=0)
            
            # Update state
            class_id = np.argmax(smoothed)
            self.current_confidence = float(smoothed[class_id])
            self.current_prediction = CLASSES[class_id]
            self.all_predictions = smoothed.tolist()
            
            # Print high confidence detections (like your original)
            current_time = time.time()
            if self.current_confidence > 0.5 and current_time - last_print_time > 0.5:
                print(f"\r🎯 {self.current_prediction}: {self.current_confidence*100:.1f}%    ", end="", flush=True)
                last_print_time = current_time
            
            # Control loop speed - matching your original timing
            time.sleep(0.05)  # 50ms like your original
        
        stream.stop()
        stream.close()

# =====================================================================
# SOCKET SERVER
# =====================================================================
class SocketServer:
    def __init__(self, processor):
        self.processor = processor
        self.server_socket = None
        
    def start(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((HOST, PORT))
            self.server_socket.listen(1)
            self.server_socket.settimeout(1.0)
            
            print(f"Server: {HOST}:{PORT}")
            print("\n✓ Ready! Connect with Streamlit app\n")
            
            while True:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"✓ Client connected: {address}")
                    
                    while True:
                        data = {
                            'prediction': self.processor.current_prediction,
                            'confidence': self.processor.current_confidence,
                            'all_predictions': self.processor.all_predictions
                        }
                        client_socket.send((json.dumps(data) + '\n').encode())
                        time.sleep(0.05)  # 20Hz update
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Connection lost: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            print(f"Server error: {e}")

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__":
    processor = AccurateAudioProcessor()
    
    if not processor.load_model():
        print("Failed to load model. Make sure the model file exists.")
        sys.exit(1)
    
    # Start audio processing
    audio_thread = threading.Thread(target=processor.process_stream, daemon=True)
    audio_thread.start()
    
    time.sleep(1)
    
    # Start server
    server = SocketServer(processor)
    server.start()