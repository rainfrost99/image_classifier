import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from workspace_utils import active_session
from PIL import Image
import numpy as np
import argparse

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def load_checkpoint(checkpoint_path):
    ### Load checkpoint file ###
    checkpoint = torch.load(checkpoint_path)
    model = models.densenet121(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im.thumbnail((256, 256))
    
    width, height = im.size
    #print('width: ', width) 
    #print('height: ', height)
    
    left = (width - 224)/2
    top = (height - 224)/2
    right = left + 224
    bottom = top + 224
    
    im = im.crop((left, top, right, bottom))
    
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    #print('orig shape: ', np_image.shape)
    
    # Transpose to put colour channel first
    np_image = np_image.transpose((2, 0, 1))
    image = torch.from_numpy(np_image)
    #print('transposed shape: ', image.shape)
        
    return image





def predict_image(model, image_path):
    
    # Image pre-processing
    image = process_image(image_path)
    
    # Get the top 5 classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    image = process_image(image_path).unsqueeze_(0).type(torch.FloatTensor).to(device)
    model.eval()
    
    with torch.no_grad():
        log_ps = model.forward(image)
        ps = torch.exp(log_ps)        
        top_p, top_class = ps.topk(5, dim=1)
   
    model.train()
    
    # Predicts image name
    
    # Get labels
    print('\n### The most likely image will be displayed first ###')
    for i, j in zip(top_class.tolist()[0], top_p.tolist()[0]):
        print(cat_to_name[str(i)], j)
    
    

def main(image_path, checkpoint_path):
    model = load_checkpoint(checkpoint_path)
    predict_image(model, image_path)


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Checkpoint_path and Image_path')

    # Add the arguments
    parser.add_argument('CheckpointPath',
                    metavar='checkpointpath',
                    type=str,
                    help='Checkpoint path to load network')

    parser.add_argument('ImagePath',
                    metavar='imagepath',
                    type=str,
                    help='Image path for prediction')

    
    # Execute the parse_args() method
    args = parser.parse_args()
    checkpoint_path = args.CheckpointPath
    image_path = args.ImagePath
    print('checkpoint_path: ', checkpoint_path)
    print('image_path: ', image_path)
    main(image_path, checkpoint_path)