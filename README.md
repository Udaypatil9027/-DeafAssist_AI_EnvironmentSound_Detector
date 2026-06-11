# 🎧 DeafAssist AI - Environmental Sound Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11+-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

**DeafAssist AI** is a real-time environmental sound detection system designed to assist individuals with hearing impairments by identifying and alerting them to important environmental sounds. The system uses deep learning to classify 10 different urban sounds from the UrbanSound8K dataset and provides visual alerts for hazardous sounds.

### 🎯 Key Features

- **Real-time Sound Detection** - Instant classification of environmental sounds
- **10 Sound Classes** - Comprehensive urban sound recognition
- **Hazard Alerts** - Visual warnings for dangerous sounds
- **Professional UI** - Modern, glass-morphism interface
- **Detection History** - Track all detections with timestamps
- **Export Functionality** - Download detection history as CSV
- **Adjustable Sensitivity** - Customizable confidence threshold
- **Low Latency** - ~30ms response time

### 🎵 Detectable Sounds

| Category | Sounds |
|----------|--------|
| **Air Conditioner** | HVAC systems, fans |
| **Car Horn** | Vehicle horns, alerts |
| **Children Playing** | Laughter, shouting, playing |
| **Dog Bark** | Barking, howling |
| **Drilling** | Power tools, construction |
| **Engine Idling** | Vehicle engines |
| **Gun Shot** | Firearms, explosions |
| **Jackhammer** | Construction equipment |
| **Siren** | Emergency vehicles |
| **Street Music** | Buskers, instruments |

### ⚠️ Hazardous Sounds (Visual Alerts)

- Car Horn
- Siren  
- Gun Shot
- Dog Bark

## 🏗️ System Architecture

┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Microphone │────▶│ Audio Backend │────▶│ TensorFlow │
│ Input │ │ (Python) │ │ Model │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │
▼ ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Streamlit │◀────│ Socket │◀────│ Sound │
│ Frontend │ │ Server │ │ Classification│
└─────────────────┘ └─────────────────┘ └─────────────────┘
│
▼
┌─────────────────┐
│ Visual │
│ Display │
└─────────────────┘


## 📦 Prerequisites

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **RAM**: Minimum 4GB (8GB recommended)
- **CPU**: Dual-core or better
- **Microphone**: Built-in or external
- **Python**: 3.8 or higher

### Required Libraries

tensorflow==2.13.0
librosa==0.10.0
sounddevice==0.4.6
streamlit==1.28.0
numpy==1.24.3
scipy==1.11.1
plotly==5.15.0
pandas==2.0.3
opencv-python==4.8.0
matplotlib==3.7.2


## 🚀 Installation Guide

Step 1: Clone or Download the Project

```bash
# Navigate to your desired directory
cd C:\Users\YourUsername\Documents

# Create project folder
mkdir DeafAssist_AI
cd DeafAssist_AI


Step 2: Set Up Virtual Environment (Recommended)

python -m venv venv
venv\Scripts\activate

Step 3: Install Dependencies

pip install tensorflow librosa sounddevice streamlit numpy scipy plotly pandas opencv-python matplotlib

Step 4: Place Your Model File
Ensure your trained model file environmental_noise_cnn.h5 is in the project root directory.

Step 5: Verify Installation

python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"

🎮 Running the Application
Method 1: Manual Run (Two Terminals)
Terminal 1 - Start Backend Server:

cd C:\Users\Asus\OneDrive\Documents\Sound_Detector
python audio_backend_accurate.py


Terminal 2 - Start Frontend:

cd C:\Users\Asus\OneDrive\Documents\Sound_Detector
streamlit run streamlit_frontend_pro.py

User Interface Overview
┌─────────────────────────────────────────────────────────────┐
│  🎧 DeafAssist AI                    ● CONNECTED           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  🎯 Sound Detected                                   │   │
│  │                                                     │   │
│  │              DOG BARK                               │   │
│  │  ████████████████████████████████ 84.2%            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │Current   │ │Confidence│ │Total     │ │Status    │      │
│  │Dog Bark  │ │84.2%     │ │47        │ │Active    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────┐    │
│  │    Confidence       │  │   Top Predictions       │    │
│  │      Gauge          │  │   ■ Dog Bark 84.2%      │    │
│  │                     │  │   ■ Car Horn 11.1%      │    │
│  └─────────────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
