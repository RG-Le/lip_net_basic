import streamlit as st
import tensorflow as tf
import os
import imageio
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to streamlit app as wide
st.set_page_config(layout="wide")

# Set Up the side bar
with st.sidebar:
    st.image("https://img.freepik.com/free-vector/artificial-intelligence-ai-robot-server-room-digital-technology-banner-computer-equipment_39422-768.jpg?w=900&t=st=1681036619~exp=1681037219~hmac=9dca911224949374328d83576414b78947f929ac3c784f1bb8204ab67ce7bcc2")
    st.title("Lip Reader")
    st.info("This is an implementation of LipNet Deep Learning Model.")


st.title("Lip Reader")
# Generating a list of options or videos
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate 2 columns
col1, col2 = st.columns(2)

if options:

    # Rendering the video
    with col1:
        st.info("The video below displays the converted video in .mp4 format")
        file_path = os.path.join('..', 'data', 's1', selected_video)
        print(file_path)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        
        # Rendering inside of the app
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


    with col2:
        st.info("This is what model sees while making prediction")
        video, annotation = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width = 400)

        st.info("This is the output of the machine learning model as tokens")
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis = 0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)
        
        st.info('Decode the raw tookens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
