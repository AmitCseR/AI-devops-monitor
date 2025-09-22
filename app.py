# Zaroori 'code ke packet' import karo
import gradio as gr
from transformers import pipeline

# Hugging Face se AI model download karo
# Yeh model photo me cheezein pehchan sakta hai
classifier = pipeline("image-classification")

# Ek function banate hain jo photo lega aur AI model se puchega
def classify_image(input_image):
    # Model photo ko process karega aur result dega
    predictions = classifier(input_image)
    # Hum result ko aache se format karke return karenge
    return {p["label"]: p["score"] for p in predictions}

# Gradio ko naya 'chehra' banane ke liye bolo
# Is baar input 'image' hoga aur output 'label' hoga
iface = gr.Interface(fn=classify_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs="label",
                     title="Amit's Imagination Engine",
                     description="Koi bhi image upload karke dekho, AI batayega ki usme kya hai.")

# App ko chalao
iface.launch()