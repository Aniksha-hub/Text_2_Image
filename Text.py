import requests
import os 
import io 
import streamlit as st 
from dotenv import load_dotenv, find_dotenv 
from PIL import Image 
from datetime import datetime 
from huggingface_hub import InferenceClient

load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY") 
client = InferenceClient(api_key="")

st.title("Text-to-Image Generator")
st.write("Generate images from text prompts using Hugging Face models.")

# Input text prompt from user
prompt = st.text_input("Enter a text prompt:")

if st.button("Generate Image"):
    try:
        # Generate image from text prompt
        with st.spinner("Generating image..."):
            image = client.text_to_image(prompt, model="black-forest-labs/FLUX.1-dev")
        
        # Display the generated image
        st.image(image, caption="Generated Image", use_container_width=True)
    except Exception as e:
        st.error(f"Error: {e}")


