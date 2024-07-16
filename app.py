import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from twilio.rest import Client
import requests
from ultralytics import YOLOv10

# Function to preprocess image for CNN model
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize image to match model's expected sizing
    img = np.array(img)  
    img = img / 255.0  # Normalize pixel values to between 0 and 1
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model's expected input shape
    return img

# Function to load CNN model
@st.cache(allow_output_mutation=True)
def load_cnn_model(model_path):
    model = load_model(model_path)
    return model

# Function to predict using CNN model
def predict_cnn(image, model):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Function to display CNN model prediction result
def display_cnn_result(prediction):
    class_names = ["smoke", "fire", "non fire"]  # Adjust based on your model's output classes
    st.subheader('CNN Model Prediction:')
    st.write(f'Class: {class_names[np.argmax(prediction)]}')
    st.write(f'Confidence: {prediction[0][np.argmax(prediction)] * 100:.2f}%')

# Function to perform real-time detection using CNN model and webcam
def process_webcam_cnn(model):
    class_names = ["smoke", "fire", "non fire"]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    frame_window = st.image([])  # Create a placeholder for the video frames
    stop_button = st.button('Stop CNN Real-time Detection')  # Create a button to stop the detection

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image.")
            break

        input_frame = Image.fromarray(frame)
        prediction = predict_cnn(input_frame, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        probability = prediction[0][class_index]

        if class_name != "non fire" and probability > 0.5:
            text = f'Class: {class_name}, Probability: {probability:.2f}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Send SMS with detection details and location
            send_sms(f'Fire or smoke detected with probability {probability:.2f}!', get_location())

        frame_window.image(frame, channels="BGR")  # Update the frame in the placeholder

    cap.release()

# Function to preprocess image for MobileNetV2 model
def preprocess_mobilenet_image(img):
    img = img.resize((224, 224))  # Resize image to match model's expected sizing
    img = np.array(img)  # Convert to numpy array
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match model's expected input shape
    return img

# Function to load MobileNetV2 model
@st.cache(allow_output_mutation=True)
def load_mobilenet_model(model_path):
    model = load_model(model_path)
    return model

# Function to predict using MobileNetV2 model
def predict_mobilenet(image, model):
    processed_img = preprocess_mobilenet_image(image)
    prediction = model.predict(processed_img)
    return prediction

# Function to display MobileNetV2 model prediction result
def display_mobilenet_result(prediction):
    class_names = ["fire", "smoke", "non fire"]  
    st.subheader('MobileNetV2 Model Prediction:')
    st.write(f'Class: {class_names[np.argmax(prediction)]}')
    st.write(f'Confidence: {prediction[0][np.argmax(prediction)] * 100:.2f}%')

# Function to perform real-time detection using MobileNetV2 model and webcam
def process_webcam_mobilenet(model):
    class_names = ["smoke", "fire", "non fire"]
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    frame_window = st.image([])  # Create a placeholder for the video frames
    stop_button = st.button('Stop MobileNetV2 Real-time Detection')  # Create a button to stop the detection

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image.")
            break

        input_frame = Image.fromarray(frame)
        prediction = predict_mobilenet(input_frame, model)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        probability = prediction[0][class_index]

        if class_name != "non fire" and probability > 0.5:
            text = f'Class: {class_name}, Probability: {probability:.2f}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Send SMS with detection details and location
            send_sms(f'Fire or smoke detected with probability {probability:.2f}!', get_location())

        frame_window.image(frame, channels="BGR")  # Update the frame in the placeholder

    cap.release()

# Function to perform real-time object detection using YOLOv10 and webcam
def process_webcam_yolo():
    model = YOLOv10("C:\\Users\\Shantha\\Desktop\\fire and smoke\\fire-and-smoke-detection\\train2\\weights\\best.pt")  # Replace with your YOLOv10 weights path
    category_dict = {0: 'fire', 1: 'smoke'}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    frame_window = st.image([])  # Create a placeholder for the video frames
    stop_button = st.button('Stop YOLOv10 Real-time Detection')  # Create a button to stop the detection

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image.")
            break

        results = model(frame)
        detections = results[0].boxes.data.cpu().numpy()

        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            category_id = int(class_id)
            label = category_dict.get(category_id, 'Unknown')
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame_window.image(frame, channels="BGR")  # Update the frame in the placeholder

        # Check if fire or smoke is detected and send SMS
        detected_classes = [category_dict.get(int(d[5]), 'Unknown') for d in detections]
        if 'fire' in detected_classes or 'smoke' in detected_classes:
            send_sms('Fire or smoke detected! ', 'get_location()')

    cap.release()

# Function to get location using IPStack API
def get_location():
    ipstack_access_key = 'ENTER THE KEY'
    url = f'http://api.ipstack.com/check?access_key={ipstack_access_key}'

    try:
        response = requests.get(url)
        data = response.json()
        lat = data['latitude']
        lng = data['longitude']
        return lat, lng
    except Exception as e:
        st.error(f"Error fetching location: {str(e)}")
        return None

# Function to send SMS using Twilio API
def send_sms(message, location=None):
    account_sid = 'ID'
    auth_token = 'TOKEN'
    twilio_phone_number = 'PHONE'
    to_phone_number = 'TO PHONE'

    if location:
        lat, lng = location
        location_url = f'https://www.google.com/maps/search/?api=1&query={lat},{lng}'
        message += f'\nLocation: {location_url}'

    client = Client(account_sid, auth_token)
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=to_phone_number
    )
# Main function for Streamlit app
def main():
    st.title('Fire and Smoke Detection App')

    # Sidebar navigation for selecting model
    st.sidebar.title('Select Model')
    model_choice = st.sidebar.radio('Choose Model:', ('CNN (Uploaded Image)', 'CNN (Real-time Detection)', 'YOLOv10 (Real-time Detection)', 'MobileNetV2 (Uploaded Image)', 'MobileNetV2 (Real-time Detection)'))

    cnn_model_path = 'C:\\Users\\Shantha\\Desktop\\fire and smoke\\new_model.h5'  
    if 'cnn_model' not in st.session_state:
        st.session_state.cnn_model = load_cnn_model(cnn_model_path)
    cnn_model = st.session_state.cnn_model

    # Load MobileNetV2 model
    mobilenet_model_path = 'C:\\Users\\Shantha\\Desktop\\fire and smoke\\mobileNetv2-model.h5' 
    if 'mobile_model' not in st.session_state:
        st.session_state.mobile_model = load_mobilenet_model(mobilenet_model_path)
    mobile_model = st.session_state.mobile_model  # Assign loaded model to mobile_model variable

    # Upload image for prediction using CNN model
    if model_choice == 'CNN (Uploaded Image)':
        st.sidebar.title('Upload Image (CNN)')
        uploaded_file = st.sidebar.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Button to trigger prediction
            if st.sidebar.button('Predict (CNN)', key='predict_button'):
                prediction = predict_cnn(image, cnn_model)
                display_cnn_result(prediction)

    # Real-time detection using CNN model
    elif model_choice == 'CNN (Real-time Detection)':
        st.header('Real-time Fire and Smoke Detection using CNN')
        st.sidebar.warning('CNN model requires webcam.')

        if st.sidebar.button('Start CNN Real-time Detection', key='cnn_real_time_button'):
            process_webcam_cnn(cnn_model)

    # Real-time object detection using YOLOv10
    elif model_choice == 'YOLOv10 (Real-time Detection)':
        st.header('Real-time Object Detection using YOLOv10')
        st.sidebar.warning('YOLOv10 model requires webcam.')

        if st.sidebar.button('Start YOLOv10 Real-time Detection', key='yolo_real_time_button'):
            process_webcam_yolo()

    # Upload image for prediction using MobileNetV2 model
    elif model_choice == 'MobileNetV2 (Uploaded Image)':
        st.sidebar.title('Upload Image (MobileNetV2)')
        uploaded_file = st.sidebar.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Button to trigger prediction
            if st.sidebar.button('Predict (MobileNetV2)', key='mobilenet_predict_button'):
                prediction = predict_mobilenet(image, mobile_model)
                display_mobilenet_result(prediction)

    # Real-time detection using MobileNetV2
    elif model_choice == 'MobileNetV2 (Real-time Detection)':
        st.header('Real-time Fire and Smoke Detection using MobileNetV2')
        st.sidebar.warning('MobileNetV2 model requires webcam.')

        if st.sidebar.button('Start MobileNetV2 Real-time Detection', key='mobilenet_real_time_button'):
            process_webcam_mobilenet(mobile_model)

if __name__ == '__main__':
    main()
