import speech_recognition as sr
from diffusers import StableDiffusionPipeline
import torch
import os
import streamlit as st
from PIL import Image

# 1. Convert audio file to text using SpeechRecognition
def voice_to_text_from_file(audio_file_path):
    recognizer = sr.Recognizer()
    
    try:
        # Load the audio file
        with sr.AudioFile(audio_file_path) as source:
            audio = recognizer.record(source)
    except Exception as e:
        st.error(f"Error loading audio file: {str(e)}")
        return None

    # Convert audio to text
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

# 2. Generate image from text using Stable Diffusion
def generate_image_from_text(prompt):
    if prompt is None:
        return None

    # Load the Stable Diffusion pipeline (ensure you're using a GPU for faster results)
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original", torch_dtype=torch.float16).to("cuda")

    # Generate image from text
    image = model(prompt).images[0]

    # Return image
    return image

# Streamlit User Interface
def main():
    st.title("Voice-to-Image Generation")
    st.markdown("Upload an audio file and get an image generated from the text.")
    
    # Add button for processing lion-lament-6926.mp3
    if st.button("Process Lion Lament Audio"):
        try:
            audio_path = os.path.join("sound", "lion-lament-6926.mp3")
            if not os.path.exists(audio_path):
                st.error("Lion Lament audio file not found in sound directory.")
                return
            
            text_prompt = voice_to_text_from_file(audio_path)
            if text_prompt:
                st.write(f"Recognized text from Lion Lament: {text_prompt}")
                image = generate_image_from_text(text_prompt)
                if image:
                    st.image(image, caption="Generated Image from Lion Lament")
                    image_path = os.path.join("output", "lion_lament_image.png")
                    os.makedirs("output", exist_ok=True)
                    image.save(image_path)
                    st.success(f"Image saved successfully at {image_path}")
                else:
                    st.write("Could not generate image from Lion Lament.")
            else:
                st.write("Could not understand the Lion Lament audio.")
        except Exception as e:
            st.error(f"Error processing Lion Lament audio: {str(e)}")

    # File uploader for audio file
    uploaded_audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

    if uploaded_audio_file:
        # Save the uploaded audio file locally
        with open("uploaded_audio.mp3", "wb") as f:
            f.write(uploaded_audio_file.getbuffer())

        # Convert audio to text
        text_prompt = voice_to_text_from_file("uploaded_audio.mp3")

        if text_prompt:
            st.write(f"Recognized text: {text_prompt}")

            # Generate image from text
            image = generate_image_from_text(text_prompt)

            if image:
                st.image(image, caption="Generated Image")
                st.save(f"generated_image.png")
            else:
                st.write("Could not generate image.")
        else:
            st.write("Could not understand the audio.")

if __name__ == "__main__":
    main()
