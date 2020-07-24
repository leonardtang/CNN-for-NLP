import torch


def test(model, device, test_loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.detach(), dim=1)
            correct += (predicted == labels).double().sum().item()
            total += labels.size(0)  # labels is a Tensor with dimension [N,1], where N is batch sample size

    testing_accuracy = 100. * correct / total
    return testing_accuracy
