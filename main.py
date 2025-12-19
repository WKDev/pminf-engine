# use zoedepth model to estimate the depth of an image
# use BiRefNet model to segment the image


from transformers import pipeline
from PIL import Image
import requests
import sys
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

PROCESSOR = 'cpu'

# load BiRefNet model
from transformers import AutoModelForImageSegmentation
birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)


# load image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)


torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to(PROCESSOR)
birefnet.eval()
birefnet.half()

def extract_object(birefnet, imagepath):
    # Data settings

    print("Transforming image...")
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to(PROCESSOR).half()
    print("Image transformed successfully")

    print("Predicting...")
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    print("Predicted successfully")

    print("Converting to PIL image...")
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    print("Image and mask combined successfully")
    return image, mask

print("Extracting object...")
image, mask = extract_object(birefnet, imagepath='original.png')
print("Object extracted successfully")

image.save('segmented_image.png')
mask.save('segmented_mask.png')




sys.exit(0)
# ##################


# load pipe
depth_estimator = pipeline(task="depth-estimation", model="Intel/zoedepth-nyu-kitti")


image.save('original.png')
# inference
outputs = depth_estimator(image)
print(outputs.keys())
depth_image = outputs['depth'] # Image of depth
image.save('original.png')
outputs['depth'].save('depth.png')
print(type(outputs['predicted_depth']))

