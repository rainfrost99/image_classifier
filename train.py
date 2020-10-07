import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from workspace_utils import active_session
from PIL import Image
import numpy as np
import argparse




def load_and_transform(args):

    # Retrieve user inputs
    data_dir = args.data_dir
    
    ### Load data, transform, dataloaders ###
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_data, valid_data, test_data, trainloader, validloader, testloader


def build_and_train(args, train_data, valid_data, test_data, trainloader, validloader, testloader):
    #### Build and train network ###
    
     # Retrieve user inputs
    arch = args.arch
    learn_rate = args.learning_rate
    epoch_num = args.epochs
    hidden_units = args.hidden_units
    gpu = args.gpu
    
    # Use GPU if available else CPU
    if gpu == True:
        # Double check if really have GPU option
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    
    
    '''
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        inputs = 1024
    elif arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        inputs = 25088
    else:
        pass
    '''
    #model = models.densenet121(pretrained=True)
    #inputs = 1024
    
    model = models.vgg16(pretrained=True)
    inputs = 25088
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(25088, 800),
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(800, 800),
                           nn.ReLU(),
                           nn.Dropout(0.1),
                           nn.Linear(800, 102),
                           nn.LogSoftmax(dim=1)
)

    model.classifier = classifier
    criterion = nn.NLLLoss()

    # Freeze feature parems, train only classifier parems
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    model.to(device)

    try:
        epochs = epoch_num
        steps = 0
        running_loss = 0
        print_every = 50
        
        for epoch in range(epochs):
            for train_images, train_labels in trainloader:
                steps += 1

                # Use GPU if available
                train_images, train_labels = train_images.to(device), train_labels.to(device)
                
                # 0 the gradients
                optimizer.zero_grad()
                
                # Get log probabilities from model
                logps = model.forward(train_images)
                
                # Calculate loss
                loss = criterion(logps, train_labels)
                
                # Backpropagation
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Keep track of training loss
                running_loss += loss.item()
                
                # Test network accuracy and loss on validation set
                if steps % print_every == 0:

                    # Put model to eval mode (Turns off dropout to use network for preds)
                    model.eval()
                    valid_loss = 0
                    accuracy = 0
                    
                    with torch.no_grad():
                        for valid_images, valid_labels in validloader:
                            
                            # Use GPU if available
                            valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)
                            
                            # Get log probabilities
                            logps = model.forward(valid_images)
                            
                            # Calculate loss and keep track
                            batch_loss = criterion(logps, valid_labels)
                            valid_loss += batch_loss.item()
                            
                            #--------------------#
                            # Calculate accuracy #
                            #--------------------#
                            ps = torch.exp(logps)
                            
                            # Returns top 1 prob and looks across all columns (dim=1)
                            top_p, top_class = ps.topk(1, dim=1)
                            
                            # Check for equality with labels and update accuracy
                            equals = top_class == valid_labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            
                        # Accuracy for ea batch
                        print(f'Epoch {epoch+1}/{epochs}.. '
                              f'Train loss: {running_loss/print_every:.3f}.. '
                              f'Valid loss: {valid_loss/len(validloader):.3f}.. '
                              f'Valid accuracy: {accuracy/len(validloader):.3f}')
                    
                    # Put model back to training mode
                    model.train()
                    
                    # Reset running loss
                    running_loss = 0
                    
        # Save checkpoint after training
        save_checkpoint(train_data, model)
        
    except Exception as e:
        print(e)
 
def save_checkpoint(train_data, model):
    #### Save checkpoint ###
    class_to_idx = train_data.class_to_idx
    model.class_to_idx = { class_to_idx[k]: k for k in class_to_idx}

    checkpoint = {'input_size': 25088,
                  'output_size':102,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                  }

    torch.save(checkpoint, 'checkpoint2.pth')    
    
    
def main(args):
    train_data, valid_data, test_data, trainloader, validloader, testloader = load_and_transform(args)
    build_and_train(args, train_data, valid_data, test_data, trainloader, validloader, testloader)
    

def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", action="store")
    parser.add_argument('--arch', action="store", dest="arch", type=str, default='densenet121')
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float)
    parser.add_argument('--epochs', action="store", dest="epochs", type=int)
    parser.add_argument('--hidden_units', action='append', dest='hidden_units', type=int)
    parser.add_argument('--gpu', action="store_true", default=False, dest='gpu')
    args = parser.parse_args()
    return args
    
    
    
if __name__ == '__main__':
    args = init_argparse()
    main(args)