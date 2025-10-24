
import gradio as gr
from transformers import pipeline
classifier = pipeline("image-classification")
def classify_image(input_image):
    predictions = classifier(input_image)
    return {p["label"]: p["score"] for p in predictions}
iface = gr.Interface(fn=classify_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs="label",
                     title="Amit's Imagination Engine",
                     description="Koi bhi image upload karke dekho, AI batayega ki usme kya hai.")
iface.launch()