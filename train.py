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
    model = models.densenet121(pretrained=True)
    inputs = 1024
    
        
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(inputs, hidden_units[0]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units[0], hidden_units[0]),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units[0], 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    criterion = nn.NLLLoss()

    # Freeze feature parems, train only classifier parems
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    model.to(device)

    # TODO: Do validation on the test set
    try: 
        epochs = epoch_num
        steps = 0
        running_loss = 0
        print_every = 50

        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1

                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    test_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(testloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(testloader):.3f}")
                    running_loss = 0
                    model.train()
        
        # Save checkpoint after training
        save_checkpoint(train_data, model)
        
    except Exception as e:
        print(e)
 
def save_checkpoint(train_data, model):
    #### Save checkpoint ###
    class_to_idx = train_data.class_to_idx
    model.class_to_idx = { class_to_idx[k]: k for k in class_to_idx}

    checkpoint = {'input_size': 1024,
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