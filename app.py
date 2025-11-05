import gradio as gr
from transformers import pipeline
import torch

# Naya, powerful model download karo jo photo banata hai
# Note: Pehli baar chalne me thoda time lega
text_to_image = pipeline("text-to-image", model="stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)

def generate_image(text):
    # Model ko bolo text se image banaye
    print("Generating image for:", text)
    result = text_to_image(text).images[0]
    print("Image generated!")
    return result

# Naya chehra (UI) banate hain
iface = gr.Interface(fn=generate_image,
                     inputs="text",
                     outputs="image",
                     title="Amit's Imagination Engine", # Ya jo bhi title aapne rakha tha
                     description="Aap kuch bhi likho (jaise 'A horse riding an astronaut on Mars'), aur AI uski photo banayega.")

iface.launch()