import streamlit as st
import tensorflow as tf
import numpy as np


def load_and_convert(input_path):
    '''
    This functions load an image from input path,
    convert it to grayscale and resize it to (28, 28)
    '''
    input_image = tf.keras.utils.load_img(
        input_path,
        color_mode='grayscale',
        target_size=(28, 28),
        interpolation='nearest',
        keep_aspect_ratio=False
    )
    return input_image


def predict(category_list, model, uploaded):
    '''
    This function classifies the image and show the result in streamlit
    '''
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
    download preapred model
    """
    model = tf.keras.models.load_model(path)
    return model 


def text_block():
    st.title("Classify Fashion Item on Image")
    st.text("")
    st.text("This is an app to predict the fashion category of the input image.")
    st.text("Caterogies: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankel boot")
    st.text("Note: Any input image will be preprocessed to a 28x28 grayscale image.")
    st.text("")


def main():
    category_list = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    model = download_model("model/mnist_fashion_saved_model")
    text_block()
    uploaded = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    st.button("Classify", on_click=predict, args=(category_list, model, uploaded))


if __name__ == '__main__':
    main()