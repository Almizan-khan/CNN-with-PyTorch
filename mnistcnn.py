import argparse
import json
import platform
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        dummy_input = torch.randn(1, 1, 28, 28)
        dummy_output = self.pool(F.relu(self.conv1(dummy_input)))
        dummy_output = self.pool(F.relu(self.conv2(dummy_output)))
        flattened_size = torch.flatten(dummy_output, 1).shape[1]

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def evaluate(model, loader, device):
    model.eval()
    total_loss, count = 0.0, 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    errors = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            targets = labels.to(device)
            total_loss += F.nll_loss(logits, targets, reduction='sum').item()
            predicted = logits.argmax(1).cpu()
            confusion += torch.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
            for target, prediction in zip(labels.tolist(), predicted.tolist()):
                if target != prediction and len(errors) < 20:
                    errors.append({'target': target, 'prediction': prediction})
            count += len(labels)
    if not count:
        raise ValueError('Evaluation requires at least one example')
    return {'loss': total_loss/count, 'accuracy': confusion.diag().sum().item()/count,
            'examples': count, 'confusion_matrix': confusion.tolist(), 'error_examples': errors}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='outputs/mnist')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--test-limit', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error('epochs, batch-size and learning-rate must be positive')
    from torchvision import datasets, transforms
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_data, val_data = random_split(full_train, [55000, 5000], generator=torch.Generator().manual_seed(args.seed))
    if args.train_limit > 0:
        train_data = Subset(train_data, range(min(args.train_limit, len(train_data))))
    if args.test_limit > 0:
        test_data = Subset(test_data, range(min(args.test_limit, len(test_data))))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_data, batch_size=256)
    test_loader = DataLoader(test_data, batch_size=256)
    model = Net().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    history, best_loss = [], float('inf')
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum, count = 0.0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = F.nll_loss(model(images.to(args.device)), labels.to(args.device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()*len(labels)
            count += len(labels)
        validation = evaluate(model, val_loader, args.device)
        row = {'epoch': epoch+1, 'train_loss': loss_sum/count, 'validation_loss': validation['loss'], 'validation_accuracy': validation['accuracy']}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation['loss'] < best_loss:
            best_loss = validation['loss']
            torch.save({'model_state': model.state_dict(), 'config': vars(args)}, destination/'model.pt')
    checkpoint = torch.load(destination/'model.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    report = {'dataset': 'MNIST', 'scope': 'subset' if args.train_limit or args.test_limit else 'full official test split',
              'train_examples': len(train_data), 'validation_examples': len(val_data), 'test': evaluate(model, test_loader, args.device),
              'history': history, 'config': vars(args), 'training_seconds': time.perf_counter()-started,
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform()},
              'selection': 'lowest validation loss; official test evaluated once after selection'}
    (destination/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
