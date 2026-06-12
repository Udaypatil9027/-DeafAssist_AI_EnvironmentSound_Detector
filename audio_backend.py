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
MODEL_PATH = 'audio_classification.keras'
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
print("DeafAssist AI - Backend")
print("="*50)

# =====================================================================
# FIXED PREPROCESSING FOR 1D MODEL
# =====================================================================
def preprocess_for_1d_model(audio_buffer):
    """
    Preprocess audio for 1D model that expects shape (None, 40)
    Your model expects flat 40-dimensional features
    """
    audio_mono = np.squeeze(audio_buffer)
    
    # Normalize
    if np.max(np.abs(audio_mono)) > 0:
        audio_mono = audio_mono / np.max(np.abs(audio_mono))
    
    # Extract MFCC features (40 coefficients)
    mfccs = librosa.feature.mfcc(
        y=audio_mono, 
        sr=SAMPLE_RATE, 
        n_mfcc=40,
        n_fft=2048,
        hop_length=512
    )
    
    # Take mean across time axis to get 40 features
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Reshape to (1, 40) for model
    return mfccs_mean.reshape(1, 40)

def preprocess_audio_alternative(audio_buffer):
    """
    Alternative: Use Mel-spectrogram and flatten
    """
    audio_mono = np.squeeze(audio_buffer)
    
    # Normalize
    if np.max(np.abs(audio_mono)) > 0:
        audio_mono = audio_mono / np.max(np.abs(audio_mono))
    
    # Generate Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=audio_mono, 
        sr=SAMPLE_RATE, 
        n_mels=40,
        n_fft=2048,
        hop_length=512
    )
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Take mean across time axis to get 40 features
    features = np.mean(spectrogram_db, axis=1)
    
    return features.reshape(1, 40)

# =====================================================================
# AUDIO PROCESSOR
# =====================================================================
class AudioProcessor:
    def __init__(self):
        self.model = None
        self.audio_queue = queue.Queue(maxsize=30)
        self.audio_buffer = np.zeros((WINDOW_SIZE, 1), dtype=np.float32)
        
        # Smoothing window
        self.prediction_history = deque(maxlen=5)
        
        # Current state
        self.current_prediction = "None"
        self.current_confidence = 0.0
        self.all_predictions = [0.0] * len(CLASSES)
        self.is_running = True
        
    def load_model(self):
        """Load the .keras model"""
        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: Model not found: {MODEL_PATH}")
            return False
        
        try:
            print("Loading model...", end=" ", flush=True)
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("✓")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            
            # Verify model expects 1D input
            if len(self.model.input_shape) == 2:
                print("✓ Model expects 1D input (good)")
            else:
                print(f"⚠️ Model expects {len(self.model.input_shape)}D input")
            
            return True
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            return
        if not self.audio_queue.full():
            self.audio_queue.put(indata.copy())
    
    def process_stream(self):
        """Main processing loop"""
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                callback=self.audio_callback,
                blocksize=1024,
                latency='low'
            )
            stream.start()
            print("🎤 Microphone active")
            print("✓ Using 1D model preprocessing\n")
        except Exception as e:
            print(f"Microphone error: {e}")
            return
        
        while self.is_running:
            # Process audio chunks
            chunks_processed = 0
            while not self.audio_queue.empty() and chunks_processed < 10:
                chunk = self.audio_queue.get()
                chunk_len = len(chunk)
                self.audio_buffer = np.roll(self.audio_buffer, -chunk_len, axis=0)
                self.audio_buffer[-chunk_len:] = chunk
                chunks_processed += 1
            
            # Preprocess for 1D model
            input_features = preprocess_for_1d_model(self.audio_buffer)
            
            # Make prediction
            predictions = self.model.predict(input_features, verbose=0)[0]
            
            # Smooth predictions
            self.prediction_history.append(predictions)
            smoothed = np.mean(self.prediction_history, axis=0)
            
            # Update state
            class_id = np.argmax(smoothed)
            self.current_confidence = float(smoothed[class_id])
            self.current_prediction = CLASSES[class_id]
            self.all_predictions = smoothed.tolist()
            
            # Print high confidence detections
            if self.current_confidence > 0.5:
                print(f"\r🎯 {self.current_prediction}: {self.current_confidence*100:.1f}%    ", end="", flush=True)
            
            time.sleep(0.05)
        
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
            
            print(f"Server listening on {HOST}:{PORT}")
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
                        time.sleep(0.05)
                        
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
    processor = AudioProcessor()
    
    if not processor.load_model():
        print("\n❌ Failed to load model. Please check:")
        print(f"   1. Model file '{MODEL_PATH}' exists")
        print(f"   2. File is a valid .keras format")
        sys.exit(1)
    
    # Start audio processing
    audio_thread = threading.Thread(target=processor.process_stream, daemon=True)
    audio_thread.start()
    
    time.sleep(1)
    
    # Start server
    server = SocketServer(processor)
    server.start()
