
import os
import csv
import numpy as np
from dotenv import load_dotenv
from tqdm.notebook import tqdm

from huggingface_hub import login
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from pricer.evaluator import evaluate
from pricer.items import Item


LITE_MODEL = False
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
N_FEATURES = 5000
VAL_SPLIT = 0.1
PATIENCE = 5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


load_dotenv(override=True)
hf_token = os.environ["HF_TOKEN"]
login(hf_token, add_to_git_credential=True)


username = "ed-donner"
dataset = f"{username}/items_lite" if LITE_MODEL else f"{username}/items_full"

# Load dataset

train, val, test = Item.from_hub(dataset)
print(f"Loaded {len(train):,} training items, {len(val):,} validation_items, {len(test):,} test items")


# Human baseline from previous testing

human_predictions = []
with open("human_in.csv", "w", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    for t in test[:100]:
       writer.writerow([t.summary, 0])


human_predictions = []
with open("human_out.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        human_predictions.append(float(row[1]))


def human_pricer(item):
    idx =test.index(item)
    return human_predictions[idx]


human = human_pricer(test[0])
actual = test[0].price
print(f"Human predicted {human} for an item that actually cost: {actual}")


evaluate(human_pricer, test, size=100)


# Log transform prices to normalize the skewed distribution

y_raw = np.array([float(item.price) for item in train])
y = np.log1p(y_raw)


print(f"Raw prices - Mean: {y_raw.mean():.2f}, Median: {np.median(y_raw):.2f}, Std: {y_raw.std():.2f}")
print(f"Log prices - Mean: {y.mean():.2f}, Median: {np.median(y):.2f}, Std: {y.std():.2f}")


# Bag of words vectorization with hashing (memory efficient, no vocab storage needed)

documents = [item.summary for item in train]
vectorizer = HashingVectorizer(n_features=N_FEATURES, stop_words="english", binary=True)
X = vectorizer.fit_transform(documents)


print(f"Features matrix shape: {X.shape}")


# NN architecture

class ImprovedPriceNueralNetwork(nn.Module):
    """
    3 hidden layer network with Batchnorm and Dropout.
    Wider first layer (256) to handle 5000 dimensional sparse input
    and then progressively narrow.
    """

    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(

            # Layer 1: Compress 5000 sparse features into dense representation
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: Refine representation
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Layer 3: Final hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output: Single price prediction (in log space)
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


# Convert to Pytorch tensors

X_tensor = torch.FloatTensor(X.toarray())
y_tensor = torch.FloatTensor(y).unsqueeze(1)


# Train / validation split (val 10% per VAL_SPLIT config value)

X_train, X_val, y_train, y_val = train_test_split(
    X_tensor, y_tensor, test_size=VAL_SPLIT, random_state=RANDOM_SEED
)

print(f"Training samples: {X_train.shape[0]:.2f}")
print(f"Validation samples: {X_val.shape[0]:.2f}")


# Dataloader for mini-back training

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# Initialize model

input_size = X_tensor.shape[1]
model = ImprovedPriceNueralNetwork(input_size)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Traininable parameters: {trainable_params:,}")


# Measure loss using HuberLoss - ore robust to outliers than MSE

loss_function = nn.HuberLoss(delta=1.0)


# Adam optimizer with weight decay for layer 2 regularization

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Cosine annealing to decay learning rate smoothly to zero

scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)


# Training loop with eary stopping

best_val_loss = float("inf")
patience_counter = 0
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Training phase
    model.train()
    epoch_losses = []

    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    avg_train_loss = np.mean(epoch_losses)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_function(val_outputs, y_val).item()
    val_losses.append(val_loss)

    # Step the learning rate schedule
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    print(
        f"Epoch [{epoch + 1}/EPOCHS] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"LR: {current_lr}:.6f"
    )

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        patience_counter = 0
        print(f"New best model saved (val_loss: {val_loss}:.4f)")
    else:
        patience_counter += 1
        print(f"No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break


# Load the best model weights

model.load_state_dict(torch.load("best_model.pt"))
print(f"\nLoaded best model with val_los: {best_val_loss:.4f}")


# Evaluation

def neural_network(item):
    """Predict price for a single item.  Returns price in original dollar scale."""
    model.eval()
    with torch.no_grad():
        vector = vectorizer.transform([item.summary])
        vector = torch.FloatTensor(vector.toarray())
        log_prediction = model(vector)[0].item()
    # Reverse the log1p transform and clamp to non negative
    return max(0, np.expm1(log_prediction))


# test

sample = test[0]
predicted = neural_network(sample)
print(f"Predicted: ${predicted:.2f} | Actual: ${sample.price:.2f}")


evaluate(neural_network, test)





