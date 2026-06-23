import timm
import torch 
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

def train_one_epoch(model, loader):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        pred = outputs.argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100 * correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)

        pred = outputs.argmax(1)

        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total


if __name__ == "__main__":
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=100   
    )

    model.cuda()

    IMG_SIZE = 224

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(IMG_SIZE, padding=16),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_ds = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_ds = datasets.CIFAR100(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50
    )
    
    for epoch in range(50):
        train_loss, train_acc = train_one_epoch(model, train_loader)
        test_acc = evaluate(model, test_loader)

        scheduler.step()

        print(
            f"Epoch {epoch+1:03d} | "
            f"Loss {train_loss:.4f} | "
            f"Train {train_acc:.2f}% | "
            f"Test {test_acc:.2f}%"
        )
    
    # save the model
    torch.save(model.state_dict(), "vit_cifar100.pth")