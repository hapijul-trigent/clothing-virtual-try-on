import streamlit as st
from PIL import Image
from src.tryon import InpaintingPipeline, load_and_process_image
from src.body_segmentation import segment_body
import torch

class TryOnApp:
    def __init__(self):
        self.pipeline = self.initialize_pipeline()
        self.segment_body = segment_body
    
    def initialize_pipeline(self):
        vae_model = "madebyollin/sdxl-vae-fp16-fix"
        pipeline_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        adapter_path = "h94/IP-Adapter"
        pipeline = InpaintingPipeline(vae_model, pipeline_model)
        pipeline.load_ip_adapter(adapter_path, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        return pipeline

    def generate_image(self, person_img, cloth_img, settings):
        _, mask_image = self.pipeline.segment_body(person_img, face=False)
        self.pipeline.set_adapter_scale(1.0)
        return self.pipeline.inpaint_image(
            prompt=settings["prompt"],
            negative_prompt=settings["negative_prompt"],
            image=person_img,
            mask_image=mask_image,
            ip_adapter_image=cloth_img,
            strength=settings["strength"],
            guidance_scale=settings["guidance_scale"],
            steps=settings["steps"]
        )

app = TryOnApp()

st.title("Virtual Clothing Try-On")

col1, col2 = st.columns(2)

with col1:
    person_image_file = st.file_uploader("Upload an image of a person", type=["jpg", "jpeg", "png"])
    if person_image_file:
        person_image = Image.open(person_image_file)
        st.image(person_image, caption="Person Image", use_column_width=True)

with col2:
    cloth_image_file = st.file_uploader("Upload an image of clothing", type=["jpg", "jpeg", "png"])
    if cloth_image_file:
        cloth_image = Image.open(cloth_image_file)
        st.image(cloth_image, caption="Clothing Image", use_column_width=True)

settings = {
    "prompt": st.text_input("Prompt", "photorealistic, perfect body, beautiful skin, realistic skin, natural skin"),
    "negative_prompt": st.text_input("Negative Prompt", "ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings"),
    "strength": st.slider("Strength", min_value=0.0, max_value=1.0, value=0.99),
    "guidance_scale": st.slider("Guidance Scale", min_value=0.0, max_value=20.0, value=7.5),
    "steps": st.slider("Inference Steps", min_value=10, max_value=200, value=100)
}

if st.button("Generate"):
    print("Ready to generate...")
    if person_image_file and cloth_image_file:
        person_img = load_and_process_image(person_image_file)
        cloth_img = load_and_process_image(cloth_image_file)
        final_image = app.generate_image(person_img, cloth_img, settings)
        st.image(final_image, caption="Person Wearing Clothing", use_column_width=True)
    else:
        st.warning("Please upload both person and clothing images.")
