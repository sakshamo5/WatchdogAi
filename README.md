# WatchDog AI

**AI-Powered Real-Time Violence Detection and Surveillance System**

---
Quick demo :https://drive.google.com/file/d/1pkC5YTBKbNAN4mZOWUIcSnn0LSuEMOm9/view?usp=sharing

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

**WatchDog AI** is an intelligent surveillance solution that leverages deep learning to detect violent activities in real-time from video feeds. The system uses a TensorFlow Lite model for rapid video analysis, provides instant WhatsApp alerts, and offers a user-friendly GUI for monitoring and incident management. Designed for enterprises, public spaces, educational institutions, and more, WatchDog AI aims to enhance safety and enable timely intervention.

---

## Features

- **Real-Time Violence Detection:** High-accuracy AI model analyzes live video feeds for violent behavior.
- **Instant Alerts:** Automated WhatsApp notifications with video evidence sent to designated recipients.
- **Automated Evidence Storage:** Captures and securely stores video clips of detected incidents.
- **User-Friendly Interface:** Tkinter-based GUI for live monitoring, settings, and alert management.
- **Customizable Settings:** Adjustable detection thresholds and camera configurations.
- **Incident Logging:** Searchable and exportable logs of all detected incidents.

---

## Architecture

- **Video Capture:** Real-time camera feed acquisition via OpenCV.
- **AI Inference:** TensorFlow Lite model processes video clips for violence detection.
- **Alert System:** Twilio API integration for WhatsApp notifications.
- **Processing:** Multi-threaded design for efficient video analysis and alerting.
- **Storage:** Organized video and incident log storage.

---

## Tech Stack

- **Programming Language:** Python
- **AI/ML:** TensorFlow Lite
- **Computer Vision:** OpenCV
- **GUI:** Tkinter
- **Notifications:** Twilio API (WhatsApp)
- **Supporting Libraries:** NumPy, Pillow (PIL), threading, requests

---

## Installation

### Prerequisites

- Python 3.8+
- [pip](https://pip.pypa.io/en/stable/)
- Webcam or IP camera access

### Clone the Repository

```bash
git clone https://github.com/[your-username]/watchdog-ai.git
cd watchdog-ai
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Additional Setup

- Download the pre-trained TensorFlow Lite model and place it in the `models/` directory.
- Obtain Twilio API credentials for WhatsApp integration.
- Configure environment variables or update `config.py` with your settings.

---

## Usage

```bash
python main.py
```

- Launches the WatchDog AI GUI.
- Start the camera feed, configure settings, and monitor alerts in real-time.

---

## Configuration

- **Camera Source:** Set camera index or RTSP stream in the settings tab or `config.py`.
- **Detection Thresholds:** Adjust sensitivity for violence detection in the GUI.
- **WhatsApp Alerts:** Enter recipient numbers and Twilio credentials in settings.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a pull request

---


## Contact

For questions, feedback:

- **Email:** [saksham05singhal@gmail.com.com]

