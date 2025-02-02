import torch
import utils
from cat_breeds.models.for_cats import SmallCNN
from torch.utils.data import DataLoader

print("Creating model...", flush=True)
device = torch.device("cuda")
model = SmallCNN()
model.to(device)

# Not using transforms because Chapel does not have transforms.

cats_train = utils.cat_breed_dataset("./example/data/catbreeds")
trainloader = DataLoader(cats_train, batch_size=128, shuffle=True)

# Define hyperparameters
print("Initializing hyperparameters...", flush=True)
epochs = 10
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

print("Starting training...", flush=True)
for epoch in range(epochs):
    utils.train(model, device, trainloader, optimizer, criterion, epoch, one_pass=False, verbose=True)

model.to(torch.device("cpu"))
torch.save(model, "./example/models/pretest.pt")