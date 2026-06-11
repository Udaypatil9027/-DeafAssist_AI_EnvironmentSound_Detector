import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import time

print("DIAGNOSING YOUR MODEL")
print("=" * 50)

# Load model
model = tf.keras.models.load_model('environmental_noise_cnn.h5')
print(f"✓ Model loaded")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")

# Test 1: Check if model produces varied outputs
print("\nTest 1: Model response to different inputs")
test_inputs = []
for i in range(5):
    random_input = np.random.randn(1, 40, 174, 1) * (i + 1) * 0.2
    pred = model.predict(random_input, verbose=0)[0]
    test_inputs.append(pred)

# Check variance
all_preds = np.array(test_inputs)
variance = np.var(all_preds, axis=0)
print(f"Average prediction variance: {np.mean(variance):.6f}")
if np.mean(variance) < 0.01:
    print("⚠️ WARNING: Model produces very similar outputs for different inputs!")
    print("   This indicates the model may not have learned properly.")

# Test 2: Record and test with real audio
print("\nTest 2: Record 2 seconds of audio for testing")
print("Make a LOUD gunshot sound or play gunshot audio now...")

duration = 2
sample_rate = 22050
recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()
print("✓ Recording complete")

# Preprocess
audio = recording.flatten()
mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=40)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

if mel_spec_db.shape[1] < 174:
    pad = 174 - mel_spec_db.shape[1]
    mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, pad)), mode='constant')
else:
    mel_spec_db = mel_spec_db[:, :174]

input_tensor = mel_spec_db.reshape(1, 40, 174, 1)
prediction = model.predict(input_tensor, verbose=0)[0]

classes = ["Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", 
           "Drilling", "Engine Idling", "Gun Shot", "Jackhammer", "Siren", "Street Music"]

print("\nPredictions for recorded sound:")
for i, (cls, prob) in enumerate(zip(classes, prediction)):
    bar = "█" * int(prob * 50)
    print(f"{i:2d}. {cls:20s}: {prob*100:5.1f}% {bar}")

top_idx = np.argmax(prediction)
print(f"\n🎯 Top prediction: {classes[top_idx]} ({prediction[top_idx]*100:.1f}%)")

if classes[top_idx] != "Gun Shot" and prediction[top_idx] < 0.3:
    print("\n⚠️ ISSUE DETECTED: Model is not detecting gunshots properly!")
    print("\nPossible solutions:")
    print("1. Your model may need retraining with more gunshot samples")
    print("2. The preprocessing might not match what the model expects")
    print("3. Try the MFCC preprocessing mode (press 'M' in the main app)")