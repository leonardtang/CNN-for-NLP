import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

def load_data():




model