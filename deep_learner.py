from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch import nn
from torch import optim
import torch
import os


def do_deep_learning(pretrained_model, data_dir, learning_rate, hidden_units, epochs, cuda=True):
    train_loader, valid_loader, test_loader, class_to_idx = __setup_data(data_dir)

    train_dir = data_dir + '/train/'
    number_of_classes = len(os.listdir(train_dir))
    number_of_inputs = pretrained_model.classifier[0].in_features

    for param in pretrained_model.parameters():
        param.requires_grad = False

    classifier = __build_classifer(number_of_inputs, number_of_classes, hidden_units)
    pretrained_model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    trained_model = __learn(pretrained_model, train_loader, valid_loader, epochs, criterion, optimizer, cuda)
    trained_model.optimizer = optimizer
    trained_model.epochs = epochs
    trained_model.class_to_idx = class_to_idx
    return trained_model


def __learn(pretrained_model, train_loader, validation_loader, epochs, criterion, optimizer, cuda):
    steps = 0
    print_every = 40
    cpu_or_cuda = 'cpu'
    if cuda:
        cpu_or_cuda = 'cuda'

    pretrained_model.to(cpu_or_cuda)

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            inputs, labels = inputs.to(cpu_or_cuda), labels.to(cpu_or_cuda)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = pretrained_model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "Running Training Loss: {:.4f}".format(running_loss / print_every))
                running_loss = 0

        count = 0
        valid_loss = 0
        for ii, (valid_inputs, valid_labels) in enumerate(validation_loader):
            count += 1
            valid_inputs, valid_labels = valid_inputs.to(cpu_or_cuda), valid_labels.to(cpu_or_cuda)
            valid_outputs = pretrained_model(valid_inputs)
            valid_loss += criterion(valid_outputs, valid_labels)

        print("Epoch {} Validation Loss: {:.4f}".format(e + 1, valid_loss / count))
        valid_accuracy = check_accuracy(validation_loader, pretrained_model, cpu_or_cuda)
        print("Epoch {} Validation Accuracy: {:.4f}%".format(e + 1, 100 * valid_accuracy))

    return pretrained_model


def check_accuracy(loader, model, cpu_or_cuda):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(cpu_or_cuda), labels.to(cpu_or_cuda)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def __build_classifer(input_count, number_of_classes, hidden_units):
    classifier = nn.Sequential(
        nn.Linear(input_count, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_units, number_of_classes),
        nn.LogSoftmax(dim=1)
    )
    return classifier


def __setup_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    validation_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_data = ImageFolder(train_dir, transform=train_transforms)
    validation_data = ImageFolder(valid_dir, transform=validation_transforms)
    test_data = ImageFolder(test_dir, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    return train_loader, validation_loader, test_loader, train_data.class_to_idx
