import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Load Real-ESRGAN pretrained model
@st.cache_resource
def load_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='RealESRGAN_x4plus.pth',
        model=model,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available()
    )
    print("Model loaded!")  # Will show in terminal, not on Streamlit
    return upsampler

# Streamlit App Setup
st.set_page_config(page_title="Image Super-Resolution", layout="centered")
st.title("🖼 Real-ESRGAN Image Super-Resolution")

uploaded_file = st.file_uploader("Upload a low-resolution image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Original Image", use_container_width=True)

    if st.button("Enhance Image"):
        with st.spinner("Upscaling in progress..."):
            upsampler = load_model()
            img = np.array(input_image)
            output, _ = upsampler.enhance(img, outscale=4)
            enhanced_image = Image.fromarray(output)
            st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
            st.success("Image successfully enhanced!")
