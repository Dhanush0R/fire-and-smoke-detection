Here's a README file for your "Fire and Smoke Detection using Computer Vision and Deep Learning Models" project:

---

# Fire and Smoke Detection using Computer Vision and Deep Learning Models

This project aims to develop a system for detecting fire and smoke in real-time using deep learning models and computer vision techniques. The system is capable of analyzing video feeds from a webcam to identify potential fire and smoke hazards, sending notifications with precise location details when a threat is detected.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Real-time Detection](#real-time-detection)
- [SMS Notifications](#sms-notifications)
- [Location Tracking](#location-tracking)
- [Acknowledgements](#acknowledgements)

## Features
- Fire and smoke detection using pre-trained CNN and MobileNetV2 models.
- Real-time object detection using YOLOv10.
- Automatic SMS notifications with location details when fire or smoke is detected.
- User-friendly web interface built with Streamlit.

## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/fire-and-smoke-detection.git
    cd fire-and-smoke-detection
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained models and place them in the specified paths:**
    - CNN model: `new_model.h5`
    - MobileNetV2 model: `mobileNetv2-model.h5`
    - YOLOv10 weights: `best.pt`

## Usage
1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Select a model and input method from the sidebar:**
    - CNN (Uploaded Image)
    - CNN (Real-time Detection)
    - YOLOv10 (Real-time Detection)
    - MobileNetV2 (Uploaded Image)
    - MobileNetV2 (Real-time Detection)

## Models
### CNN Model
- Trained to detect smoke, fire, and non-fire images.
- Usage:
    - Upload an image or use the webcam for real-time detection.

### MobileNetV2 Model
- Lightweight and efficient model for mobile and edge devices.
- Usage:
    - Upload an image or use the webcam for real-time detection.

### YOLOv10 Model
- Advanced object detection model for real-time applications.
- Usage:
    - Use the webcam for real-time detection.

## Real-time Detection
- **Webcam Access:**
    - Ensure your webcam is properly connected and accessible.
    - Start the real-time detection by selecting the appropriate option from the sidebar.

## SMS Notifications
- **Twilio API:**
    - Automatically sends an SMS when fire or smoke is detected.
    - Configure Twilio credentials in the `send_sms` function:
      ```python
      account_sid = 'your_twilio_account_sid'
      auth_token = 'your_twilio_auth_token'
      twilio_phone_number = 'your_twilio_phone_number'
      to_phone_number = 'destination_phone_number'
      ```

## Location Tracking
- **IPStack API:**
    - Retrieves the current location based on IP address.
    - Configure IPStack access key in the `get_location` function:
      ```python
      ipstack_access_key = 'your_ipstack_access_key'
      ```

## Acknowledgements
- This project utilizes several open-source libraries and pre-trained models.
- Special thanks to the developers and researchers who contributed to the creation of these tools and datasets.

---

### Notes
- Ensure your webcam drivers are up to date.
- Properly configure the paths to your model files and API keys before running the application.

---

