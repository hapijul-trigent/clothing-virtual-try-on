import streamlit as st
from PIL import Image
from src.tryon import InpaintingPipeline, load_and_process_image
from src.SegBody import segment_body
import torch
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image


# Set main panel
# favicon = Image.open("static/images/Trigent_Logo.png")
st.set_page_config(
    page_title="Trigent VTON | Trigent AXLR8 Labs",
    page_icon=':camera:',
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Add logo and title
logo_path = "https://trigent.com/wp-content/uploads/Trigent_Axlr8_Labs.png"
st.markdown(
    f"""
    <div style="text-align: center;">
        <img src="{logo_path}" alt="Trigent Logo" style="max-width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)
# Main Page Title and Caption
st.title("Trigent - Virtual Try-On")
st.caption("Streamlit application that leverages a pre-trained model from Hugging Face to provide a virtual clothing try-on feature.")
st.divider()

def get_or_create_session_state_variable(key, default_value=None):
    """
    Retrieves the value of a variable from Streamlit's session state.
    If the variable doesn't exist, it creates it with the provided default value.

    Args:
        key (str): The key of the variable in session state.
        default_value (Any): The default value to assign if the variable doesn't exist.

    Returns:
        Any: The value of the session state variable.
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]



get_or_create_session_state_variable(key='TryOnApp', default_value=False)
# if not st.session_state['TryOnApp']:
# app = loadTryOnApp()
# st.session_state['TryOnApp'] = True
@st.cache_resource()
def virtual_try_on_pipeline(ip_scale=1.0):
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to("cuda")
    pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", low_cpu_mem_usage=True)
    pipeline.set_ip_adapter_scale(ip_scale)
    return pipeline


def virtual_try_on(pipeline, img, clothing, prompt, negative_prompt, ip_scale=1.0, strength=0.99, guidance_scale=7.5, steps=100):
    _, mask_img = segment_body(img, face=False)
    pipeline.set_ip_adapter_scale(ip_scale)
    images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        mask_image=mask_img,
        ip_adapter_image=clothing,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
    ).images
    return images[0]



vton_pipeline = virtual_try_on_pipeline()

col1, col2, col3 = st.columns(3)

with col1:
    person_image_file = st.file_uploader("Upload an image of a person", type=["jpg", "jpeg", "png"])
    if person_image_file:
        person_image = Image.open(person_image_file)
        st.image(person_image, caption="Person Image", use_column_width=True)
        person_image.save('person.jpg')
with col2:
    cloth_image_file = st.file_uploader("Upload an image of clothing", type=["jpg", "jpeg", "png"])
    if cloth_image_file:
        cloth_image = Image.open(cloth_image_file)
        st.image(cloth_image, caption="Clothing Image", use_column_width=True)
        cloth_image.save('cloth.jpg')
settings = {
    "prompt": st.text_input("Prompt", "photorealistic, perfect body, beautiful skin, realistic skin, natural skin"),
    "negative_prompt": st.text_input("Negative Prompt", "ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings"),
    "strength": st.slider("Strength", min_value=0.0, max_value=1.0, value=0.99),
    "guidance_scale": st.slider("Guidance Scale", min_value=0.0, max_value=20.0, value=7.5),
    "steps": st.slider("Inference Steps", min_value=10, max_value=200, value=100)
}

with col3:
  st.subheader('')

  if st.button("Try On Cloth", use_container_width=True):
      print("Ready to generate...")
      if person_image_file and cloth_image_file:
          person_img = load_image('person.jpg').convert("RGB")
          cloth_img = load_image('cloth.jpg').convert("RGB")
          # final_image = app.generate_image(person_img, cloth_img, body_mask_image=body_mask_image, settings=settings)
          st.subheader('')
          with st.spinner(text='Fashion is the armor to survive the reality of everyday life. — Bill Cunningham'):
              generated_image = virtual_try_on(
                  vton_pipeline, img=person_img, clothing=cloth_img, 
                  prompt=settings['prompt'], negative_prompt=settings['negative_prompt'], 
                  ip_scale=1.0, strength=settings['strength'], guidance_scale=settings['guidance_scale'], steps=settings['steps']
              )
              with col3:
              
                st.image(generated_image, caption="Person Wearing Clothing", use_column_width=True)
      else:
          st.warning("Please upload both person and clothing images.")


# Footer with Font Awesome icons
footer_html = """
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
<div style="text-align: center; margin-right: 10%;">
    <p>
        &copy; 2024, Trigent Software Inc. All rights reserved. |
        <a href="https://www.linkedin.com/company/trigent-software" target="_blank" aria-label="LinkedIn"><i class="fab fa-linkedin"></i></a> |
        <a href="https://www.twitter.com/trigent-software" target="_blank" aria-label="Twitter"><i class="fab fa-twitter"></i></a> |
        <a href="https://www.youtube.com/trigent-software" target="_blank" aria-label="YouTube"><i class="fab fa-youtube"></i></a>
    </p>
</div>
"""
# Custom CSS to make the footer sticky
footer_css = """
<style>
.footer {
    position: fixed;
    z-index: 1000;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    color: black;
    text-align: center;
}
[data-testid="stSidebarNavItems"] {
    max-height: 100%!important;
}
</style>
"""
# Combining the HTML and CSS
footer = f"{footer_css}<div class='footer'>{footer_html}</div>"
# Rendering the footer
st.markdown(footer, unsafe_allow_html=True)