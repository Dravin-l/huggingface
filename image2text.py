from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

load_dotenv(find_dotenv())

def image2text(url):
    imagetext = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text=imagetext(url)[0]["generated_text"]

    print(text)
    return text

image2text("3037904.png")