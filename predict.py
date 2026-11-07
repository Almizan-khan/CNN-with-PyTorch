import argparse
import json
import torch
from PIL import Image, ImageOps
from mnistcnn import Net

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('image')
    parser.add_argument('--invert', action='store_true')
    args = parser.parse_args()
    from torchvision.transforms.functional import to_tensor
    model = Net()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True)['model_state'])
    model.eval()
    image = Image.open(args.image).convert('L').resize((28, 28))
    if args.invert:
        image = ImageOps.invert(image)
    tensor = (to_tensor(image)-0.1307)/0.3081
    with torch.inference_mode():
        probabilities = model(tensor.unsqueeze(0)).exp()[0]
    print(json.dumps({'digit': probabilities.argmax().item(), 'probabilities': probabilities.tolist()}))

if __name__ == '__main__':
    main()
