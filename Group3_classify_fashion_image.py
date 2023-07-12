import streamlit as st
import tensorflow as tf
import numpy as np


def load_and_convert(input_path):
    """
    This function loads an image from the input path,
    converts it to grayscale, and resizes it to (28, 28).
    """
    input_image = tf.keras.utils.load_img(
        input_path,
        color_mode='grayscale',
        target_size=(28, 28),
        interpolation='nearest',
        keep_aspect_ratio=False
    )
    return input_image


def predict(category_list, model, uploaded):
    """
    This function classifies the image and shows the result in Streamlit.
    """
    if uploaded:
        input_image = load_and_convert(uploaded)
        st.image(input_image, caption="Preprocessed Image (28x28 grayscale)")
        input_array = tf.keras.utils.img_to_array(input_image)
        pred = model.predict(np.expand_dims(input_array, axis=0), verbose=0)
        pred = np.argmax(pred, axis=-1).squeeze()
        cat = category_list[pred]
        st.subheader(f"It looks like a :blue[{cat}]")
    else:
        st.error("No image has been uploaded")


def download_model(path):
    """
    Download the pre-trained model.
    """
    model = tf.keras.models.load_model(path)
    return model 


def text_block():
    st.title("Classify Fashion Item from Image")
    st.text("")
    st.text("This app predicts the fashion category of an input image.")
    st.text("Categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot")
    st.text("Note: The input image will be preprocessed as a 28x28 grayscale image.")
    st.text("")


def main():
    category_list = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    model = download_model("model/mnist_fashion_saved_model")
    text_block()
    uploaded = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    st.button("Classify", on_click=predict, args=(category_list, model, uploaded))


if __name__ == '__main__':
    main()
