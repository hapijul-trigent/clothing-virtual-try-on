import spaces
from diffusers import AutoPipelineForInpainting, AutoencoderKL
from diffusers.utils import load_image
import torch
import matplotlib.pyplot as plt


class InpaintingPipeline:
    def __init__(self, vae_model, pipeline_model, device="cuda"):
        self.device = device
        self.vae = self.load_vae(vae_model)
        self.pipeline = self.load_pipeline(pipeline_model)
    
    def load_vae(self, model_name):
        return AutoencoderKL.from_pretrained(model_name, torch_dtype=torch.float16)
    
    def load_pipeline(self, model_name):
        return AutoPipelineForInpainting.from_pretrained(
            model_name,
            vae=self.vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

    def load_ip_adapter(self, adapter_path, subfolder, weight_name):
        self.pipeline.load_ip_adapter(adapter_path, subfolder=subfolder, weight_name=weight_name, low_cpu_mem_usage=True)
    
    def set_adapter_scale(self, scale):
        self.pipeline.set_ip_adapter_scale(scale)
    
    def inpaint_image(self, prompt, negative_prompt, image, mask_image, ip_adapter_image, strength, guidance_scale, steps):
        return self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            ip_adapter_image=ip_adapter_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        ).images[0]

def load_and_process_image(image_path):
    return load_image(image_path).convert("RGB")

# def main():
#     vae_model = "madebyollin/sdxl-vae-fp16-fix"
#     pipeline_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
#     adapter_path = "h94/IP-Adapter"
    
#     inpainting_pipeline = InpaintingPipeline(vae_model, pipeline_model)
#     inpainting_pipeline.load_ip_adapter(adapter_path, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
    
#     image = load_and_process_image('/content/example_human_01992_00.jpg')
#     ip_image = load_and_process_image('/content/example_cloth_09163_00.jpg')
    
#     seg_image, mask_image = segment_body(image, face=False)
    
#     inpainting_pipeline.set_adapter_scale(1.0)
    
#     final_image = inpainting_pipeline.inpaint_image(
#         prompt="photorealistic, perfect body, beautiful skin, realistic skin, natural skin",
#         negative_prompt="ugly, bad quality, bad anatomy, deformed body, deformed hands, deformed feet, deformed face, deformed clothing, deformed skin, bad skin, leggings, tights, stockings",
#         image=image,
#         mask_image=mask_image,
#         ip_adapter_image=ip_image,
#         strength=0.99,
#         guidance_scale=7.5,
#         steps=100
#     )
    
#     plt.imshow(final_image)
#     plt.show()

# if __name__ == "__main__":
#     main()



class TryOnApp:
    def __init__(self):
        self.pipeline = self.initialize_pipeline()
    
    def initialize_pipeline(self):
        vae_model = "madebyollin/sdxl-vae-fp16-fix"
        pipeline_model = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
        adapter_path = "h94/IP-Adapter"
        pipeline = InpaintingPipeline(vae_model, pipeline_model)
        pipeline.load_ip_adapter(adapter_path, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
        return pipeline

    def generate_image(self, person_img, cloth_img, body_mask_image, settings):
        _, mask_image = body_mask_image
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


@st.cache_resource()
def loadTryOnApp(): return TryOnApp()


if __name__ == "__main__":
    main()
