import time
import copy
import torch
from torch import nn
from torch.autograd import Variable


def train(model, device, train_loader, val_loader, batch_size, n_epochs=20, learning_rate=1.0):

    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("-" * 30)

    # Adadelta with L2 regularization (weight_decay=0)
    params_to_update = model.parameters()
    optimizer = torch.optim.Adadelta(params_to_update, lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)

    since = time.time()
    n_batches = len(train_loader)
    print_every = n_batches // 10
    val_loss_history = []
    val_acc_history = []
    train_loss_history = []
    train_acc_history = []
    print("Pre copy")
    model_weights = copy.deepcopy(model.state_dict())
    print("Post copy")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print("Post model ad hoc stuff")

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch + 1, n_epochs))
        print('-' * 10)

        model.train()
        batch_loss = 0.0
        training_loss = 0.0
        training_corrects = 0

        # One loop through training set
        print("Pre-for loop")
        for i, data in enumerate(train_loader, 0):
            print("Entering for loop")
            inputs, labels = data
            print(inputs)
            print(inputs.size())
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            print("Data moved to GPU")
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.detach(), dim=1)
            loss = criterion(outputs, labels)
            batch_loss += loss.item()
            training_corrects += (predicted == labels).double().sum().item()

            loss.backward()
            optimizer.step()

            # Print average batch loss and time elapsed after every 10 mini-batches
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch %d, %d %% \t Average Batch Loss: %.2f took: %.2fs" % (
                    epoch + 1, int(100 * (i + 1) / n_batches), batch_loss / print_every, time.time() - since))
                # Reset running batch loss
                training_loss += batch_loss
                batch_loss = 0.0

        average_train_loss = training_loss / n_batches
        print("Average Training Loss Per Batch: %.2f" % average_train_loss)
        average_train_accuracy = 100 * training_corrects / (n_batches * batch_size)
        print("Training Accuracy: %.2f %%" % average_train_accuracy)
        train_loss_history.append(average_train_loss)
        train_acc_history.append(average_train_accuracy)

        # At the end of the epoch, do a pass on the validation set
        model.eval()
        running_val_loss = 0
        val_accuracy = 0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = Variable(inputs), Variable(labels)
                inputs, labels = inputs.to(device), labels.to(device)

                val_outputs = model(inputs)
                val_loss = criterion(val_outputs, labels)
                running_val_loss += val_loss.item()
                val_loss_history.append(val_loss)

                _, predicted = torch.max(val_outputs.detach(), dim=1)
                val_corrects += (predicted == labels).double().sum().item()

            average_val_accuracy = 100 * val_corrects / (len(val_loader) * batch_size)
            average_val_loss = running_val_loss / len(val_loader)
            val_loss_history.append(average_val_loss)
            val_acc_history.append(val_accuracy)

            print("Validation Loss: %.2f" % average_val_loss)
            print("Validation Accuracy: %.2f %%" % average_val_accuracy)

    print("Training finished in %.2f" % (time.time() - since))

    return model_weights, train_loss_history, train_acc_history, val_loss_history, val_acc_history
