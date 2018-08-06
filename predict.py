from argparse import ArgumentParser
from torchvision import models
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torch
import json


def main():
    args = get_args()
    with open(args.category_name_map) as file:
        category_name_map = json.load(file)
        
    model = __load_model(args.checkpoint, args.use_gpu)
    
    values, classes = predict(args.path_to_image, model, args.top_k, args.use_gpu)
    __make_prediction(values,classes,category_name_map)


def __make_prediction(values, classes, cat_to_name):
    print("Top {} probabilities:".format(len(values)))
    for i, value in enumerate(values):    
        flower = cat_to_name[str(classes[i])]
        print("{:.1f}% {}".format(value*100,flower))


def __load_model(filepath, gpu):

    checkpoint = torch.load(filepath) if gpu else torch.load(filepath, map_location='cpu')

    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.eval()
    return model


def predict(image_path, model, topk, gpu):
    cuda_or_cpu = 'cpu'
    if gpu:
        cuda_or_cpu = 'cuda'
        
    index_to_class = dict([[v,k] for k,v in model.class_to_idx.items()])
    image = process_image(image_path)
    torch_image = torch.from_numpy(np.expand_dims(image, axis=0)).double().to(cuda_or_cpu)
    model.double().to(cuda_or_cpu)
    output = model(torch_image)
    class_probs = F.softmax(output, dim=1)
    values, indices = torch.topk(class_probs,topk,1)

    classes = []
    for inx in indices[0]:
        classes.append(index_to_class[inx.tolist()])

    return values[0].tolist(), classes


def process_image(image, crop=True):
    im = Image.open(image)
    dims = (256,256)
    im.thumbnail(dims)
    left = (256 - 224)/2
    right = (256 + 224)/2
    upper = (256 - 224)/2
    lower = (256 + 224)/2
    crop_zone = (left, upper, right, lower)
    if crop:
        im = im.crop(crop_zone)
    np_image = np.array(im)
    np_image = np_image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image


def get_args():
    parser = ArgumentParser()
    parser.add_argument('path_to_image', type=str, help="Path to image")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint to load")
    parser.add_argument("-k", "--top_K", dest="top_k", type=int, help="Show top K predictions", default=3)
    parser.add_argument("-c", "--category_names", dest="category_name_map", help="Map of categories to names", type=str),
    parser.add_argument("-g", "--gpu", dest="use_gpu", help="Use gpu for prediction", type=bool, default=False)
    return parser.parse_args()

main()