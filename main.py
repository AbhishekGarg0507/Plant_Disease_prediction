import streamlit as st
import tensorflow as tf
import numpy as np

# tenserflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('Trained_Model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    # converting image to batch
    input_array = np.array([input_array]) 
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    return result_index

# sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select page", ["Home", "About","Disease Recognition"])

# home page
if app_mode == "Home":
    st.header("PLANT DISEASE DETECTION SYSTEM")
    image_pth = "home_image.jpg"
    st.image(image_pth, caption="Plant Disease Detection System", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.

    """)

# about page
elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    ### Dataset Overview
    This dataset is recreated using offline augmentation from the original dataset. 
    The original dataset can be found on this github repo. 
    This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.
    The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    A new directory containing 33 test images is created later for prediction purpose.

    #### Content
    1. Train (70295 images)
    2. Validation (17572 images)
    3. Test (33 images)

    ### Future scope 
    - The model can be further improved by using more complex architectures and techniques such as transfer learning, data augmentation, and hyperparameter tuning.
    - The system can be extended to include more plant species and diseases, making it more versatile and useful for a wider range of users.
    - Integration with mobile applications can allow farmers and gardeners to easily access the disease recognition system in the field, providing real-time assistance and improving crop management practices.
    """)

# disease recognition page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])
    
    if(st.button("Show Image")):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True) 

    if(st.button("Predict Disease")):
        with st.spinner("Predicting..."):
            result_index = model_prediction(uploaded_file)

            # mapping the result index to the corresponding disease name
            class_name = ['Apple___Apple_scab',
                        'Apple___Black_rot',
                        'Apple___Cedar_apple_rust',
                        'Apple___healthy',
                        'Blueberry___healthy',
                        'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___healthy',
                        'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)',
                        'Peach___Bacterial_spot',
                        'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot',
                        'Pepper,_bell___healthy',
                        'Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy',
                        'Raspberry___healthy',
                        'Soybean___healthy',
                        'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch',
                        'Strawberry___healthy',
                        'Tomato___Bacterial_spot',
                        'Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
        st.success(f"The predicted disease is: {class_name[result_index]}")
