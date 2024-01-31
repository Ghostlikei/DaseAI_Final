from PIL import Image
import requests
from transformers import AutoProcessor, FlavaModel

model = FlavaModel.from_pretrained("facebook/flava-full")
processor = AutoProcessor.from_pretrained("facebook/flava-full")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

print(image)

inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True)

# print(inputs)

outputs = model(**inputs)

print("Config of model: ", model.config)

image_embeddings = outputs.image_embeddings
text_embeddings = outputs.text_embeddings
multimodal_embeddings = outputs.multimodal_embeddings

# print(outputs.last_hidden_state.shape)

# print(text_embeddings.shape)

# print(multimodal_embeddings.shape)