import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#from workspace_utils import active_session
from PIL import Image
import numpy as np
import argparse


def load_and_transform(data_dir):
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
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_data, test_data, trainloader, testloader

def build_and_train(train_data, test_data, trainloader, testloader):
    #### Build and train network ###
    
    # Use GPU if available else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(nn.Linear(1024, 600),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(600, 600),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(600, 102),
                               nn.LogSoftmax(dim=1))
    model.classifier = classifier
    criterion = nn.NLLLoss()

    # Freeze feature parems, train only classifier parems
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)

    # TODO: Do validation on the test set
    try: 
        epochs = 5
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
                        for inputs, labels in testloader:
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
    
    
def main(input_path):
    train_data, test_data, trainloader, testloader = load_and_transform(input_path)
    build_and_train(train_data, test_data, trainloader, testloader)
    
    
if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Datapath')

    # Add the arguments
    parser.add_argument('Path',
                    metavar='path',
                    type=str,
                    help='the path to dataset')

    # Execute the parse_args() method
    args = parser.parse_args()
    input_path = args.Path
    main(input_path)