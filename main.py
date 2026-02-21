import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import json

with open('disease_info.json', 'r') as f:
    disease_info = json.load(f)

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

# tenserflow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_array = tf.keras.preprocessing.image.img_to_array(image)
    # converting image to batch
    input_array = np.array([input_array]) 
    prediction = model.predict(input_array)
    result_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return result_index, confidence,prediction[0]

# sidebar
st.sidebar.image("logo.png", width=50)  # add a logo
st.sidebar.title("Plant Disease Detector")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox("Navigate", ["Home", "About","Disease Recognition"])
st.sidebar.markdown("---")
st.sidebar.info("📸 Upload a clear, well-lit photo of a **single leaf** for best results.")
st.sidebar.markdown("**Model accuracy:** ~95%")
st.sidebar.markdown("**Supported crops:** 14 species, 38 classes")

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
    > Click **Disease Recognition** in the sidebar to get started!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.

    """)

# about page
elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    ### Dataset Overview
    This model was trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset),
    recreated using offline augmentation. It contains ~87,000 RGB images of healthy and diseased crop leaves
    categorized into **38 classes** across **14 plant species**.

    | Split | Images |
    |-------|--------|
    | Train | 70,295 |
    | Validation | 17,572 |
    | Test | 33 |

    ### Model Architecture
    A custom CNN with 5 convolutional blocks (32→512 filters), max pooling, dropout regularization,
    and a 1500-unit dense hidden layer. Trained with Adam optimizer for 6 epochs.

    ### Future Scope
    - Transfer learning with EfficientNet or MobileNetV2 for higher accuracy
    - Extended species and disease coverage
    - Mobile app integration for real-time field use
    """)

    st.markdown("---")
    st.subheader("📈 Training History")

    with open('training_history.json', 'r') as f:
        history = json.load(f)

    epochs = list(range(1, len(history['accuracy']) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'],    mode='lines+markers', name='Train Accuracy',    line=dict(color='#2e7d32', width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'],mode='lines+markers', name='Val Accuracy',      line=dict(color='#81c784', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=epochs, y=history['loss'],        mode='lines+markers', name='Train Loss',        line=dict(color='#c62828', width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'],    mode='lines+markers', name='Val Loss',          line=dict(color='#ef9a9a', width=2, dash='dash')))
    fig.update_layout(
        title="Accuracy & Loss over Epochs",
        xaxis_title="Epoch",
        yaxis_title="Value",
        plot_bgcolor="#f0f7f0",
        paper_bgcolor="#f0f7f0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

# disease recognition page
elif app_mode == "Disease Recognition":
    st.header("🔍 Disease Recognition")

    # Camera or upload choice
    img_source = st.radio("Choose input method", ["📁 Upload Image", "📷 Use Camera"], horizontal=True)

    if img_source == "📁 Upload Image":
        uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    else:
        uploaded_file = st.camera_input("Take a photo of the leaf")

    if uploaded_file is not None:

        # ── Show uploaded image (single column, constrained width) ──────────
        st.image(uploaded_file, caption="Uploaded Leaf Image", width=400)
        st.markdown("---")
        # ────────────────────────────────────────────────────────────────────

        if st.button("🔬 Predict Disease"):
            with st.spinner("Analyzing leaf..."):
                result_index, confidence, full_pred = model_prediction(uploaded_file)

            # ── CONFIDENCE THRESHOLD CHECK ───────────────────────────────────────
            # If confidence is too low, the image is likely not a valid plant leaf
            CONFIDENCE_THRESHOLD = 50  # you can tune this value (50% is a good starting point)

            if confidence < CONFIDENCE_THRESHOLD:
                st.error("⚠️ Could not confidently identify a plant disease.")
                st.warning(f"Confidence too low ({confidence:.1f}%). Please upload a **clear, well-lit photo of a single plant leaf** and try again.")
                st.stop()  # stops the rest of the results from rendering
            # ────────────────────────────────────────────────────────────────────    

            predicted_class = class_name[result_index]
            is_healthy = "healthy" in predicted_class.lower()
            display_name = predicted_class.replace("___", " → ").replace("_", " ")

            # ── PREDICTION RESULT CARD ───────────────────────────────────────
            bg_color = "#e8f5e9" if is_healthy else "#ffebee"
            border_color = "#2e7d32" if is_healthy else "#c62828"
            icon = "✅" if is_healthy else "⚠️"

            st.markdown(f"""
                <div class="result-card" style="background:{bg_color}; border: 2px solid {border_color};">
                    <h2>{icon} {display_name}</h2>
                </div>
            """, unsafe_allow_html=True)
            # ────────────────────────────────────────────────────────────────

            st.markdown("---")

            # ── CONFIDENCE METER ─────────────────────────────────────────────
            st.markdown("#### 📊 Confidence Score")
            if confidence >= 80:
                conf_label = "High confidence"
            elif confidence >= 60:
                conf_label = "Moderate confidence"
            else:
                conf_label = "Low confidence — try a clearer image"

            st.metric(label=conf_label, value=f"{confidence:.1f}%")
            st.progress(int(confidence))
            # ────────────────────────────────────────────────────────────────

            st.markdown("---")

            # ── TOP 3 PREDICTIONS ────────────────────────────────────────────
            st.markdown("#### 🏆 Top 3 Predictions")
            top3_indices = np.argsort(full_pred)[-3:][::-1]
            for idx in top3_indices:
                name = class_name[idx].replace("___", " → ").replace("_", " ")
                score = full_pred[idx] * 100
                st.write(f"**{name}** — {score:.1f}%")
                st.progress(int(score))
            # ────────────────────────────────────────────────────────────────

            st.markdown("---")

            # ── DISEASE INFO CARD ────────────────────────────────────────────
            # Loads info from disease_info.json using the predicted class name as key
            if predicted_class in disease_info:
                info = disease_info[predicted_class]
                severity = info['severity']
                severity_icons = {"None": "✅", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
                severity_icon = severity_icons.get(severity, "⚪")
                card_class = "info-card" if is_healthy else "info-card info-card-red"

                st.markdown("#### 📋 Disease Information")
                st.markdown(f"""
                    <div class="{card_class}">
                        <p><b>Severity:</b> {severity_icon} {severity}</p>
                        <p><b>Cause:</b> {info['cause']}</p>
                        <p><b>Symptoms:</b> {info['symptoms']}</p>
                        <p><b>Treatment:</b> {info['treatment']}</p>
                        <p><b>Prevention:</b> {info['prevention']}</p>
                    </div>
                """, unsafe_allow_html=True)
            # ────────────────────────────────────────────────────────────────
    else:
        st.info("👆 Please upload an image or use the camera to get started.")











