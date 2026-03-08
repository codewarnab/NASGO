#!/usr/bin/env python3
"""
PyTorch training script for Neural Architecture Search.

This script receives an architecture definition as JSON, builds the corresponding
PyTorch model, trains it on the specified dataset, and outputs results as JSON.

Usage:
    python train.py --arch arch.json --dataset cifar10 --epochs 50 --output-format json

The architecture JSON follows the NASNet cell-based format where:
- Normal cells preserve spatial dimensions
- Reduction cells halve spatial dimensions
- Cells are stacked to form the full network

Dependencies:
    pip install torch torchvision
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
except ImportError:
    print(json.dumps({
        "error": "PyTorch not installed. Run: pip install torch torchvision",
        "accuracy": 0,
        "validation_accuracy": 0,
    }))
    sys.exit(1)


# ─── Operation Registry ─────────────────────────────────────────────────────

OPERATIONS = {
    "none": lambda c, stride: Zero(stride),
    "identity": lambda c, stride: Identity() if stride == 1 else FactorizedReduce(c, c),
    "conv_1x1": lambda c, stride: ConvBNReLU(c, c, 1, stride, 0),
    "conv_3x3": lambda c, stride: ConvBNReLU(c, c, 3, stride, 1),
    "conv_5x5": lambda c, stride: ConvBNReLU(c, c, 5, stride, 2),
    "conv_7x7": lambda c, stride: ConvBNReLU(c, c, 7, stride, 3),
    "sep_conv_3x3": lambda c, stride: SepConv(c, c, 3, stride, 1),
    "sep_conv_5x5": lambda c, stride: SepConv(c, c, 5, stride, 2),
    "dil_conv_3x3": lambda c, stride: DilConv(c, c, 3, stride, 2, 2),
    "dil_conv_5x5": lambda c, stride: DilConv(c, c, 5, stride, 4, 2),
    "max_pool_3x3": lambda c, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    "avg_pool_3x3": lambda c, stride: nn.AvgPool2d(3, stride=stride, padding=1),
    "zero": lambda c, stride: Zero(stride),
}

# Map integer operation codes to string names
OP_NAMES = [
    "none", "identity", "conv_1x1", "conv_3x3", "conv_5x5",
    "conv_7x7", "sep_conv_3x3", "sep_conv_5x5", "dil_conv_3x3",
    "dil_conv_5x5", "max_pool_3x3", "avg_pool_3x3", "zero",
]


# ─── Building Blocks ────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Standard Conv-BatchNorm-ReLU block."""

    def __init__(self, c_in, c_out, kernel, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, c_in, c_out, kernel, stride, padding):
        super().__init__()
        self.op = nn.Sequential(
            # Depthwise
            nn.Conv2d(c_in, c_in, kernel, stride=stride, padding=padding, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            # Second depthwise (following DARTS convention)
            nn.Conv2d(c_out, c_out, kernel, stride=1, padding=padding, groups=c_out, bias=False),
            nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    """Dilated (atrous) convolution."""

    def __init__(self, c_in, c_out, kernel, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel, stride=stride, padding=padding,
                      dilation=dilation, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    """Identity skip connection."""

    def forward(self, x):
        return x


class Zero(nn.Module):
    """Zero operation: outputs zeros (effectively removes the edge)."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0
        return x[:, :, ::self.stride, ::self.stride] * 0.0


class FactorizedReduce(nn.Module):
    """Reduce spatial dimensions by 2x while preserving channels."""

    def __init__(self, c_in, c_out):
        super().__init__()
        assert c_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_out // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)


# ─── Cell and Network ────────────────────────────────────────────────────────

class Cell(nn.Module):
    """A single cell in the network, built from the architecture specification."""

    def __init__(self, cell_spec, channels, reduction=False):
        super().__init__()
        self.reduction = reduction
        stride = 2 if reduction else 1
        self.num_input_nodes = cell_spec.get("num_input_nodes", 2)

        # Preprocessing: adjust channel dimensions of inputs
        self.preprocess = nn.ModuleList()
        for _ in range(self.num_input_nodes):
            self.preprocess.append(ConvBNReLU(channels, channels, 1, 1, 0))

        # Build nodes from spec
        self.nodes = nn.ModuleList()
        self.edge_info = []

        for node_spec in cell_spec["nodes"]:
            node_ops = nn.ModuleList()
            node_edges = []
            for edge in node_spec["edges"]:
                op_idx = edge["operation"]
                op_name = OP_NAMES[op_idx] if isinstance(op_idx, int) else op_idx

                # Apply stride only for edges from input nodes in reduction cells
                edge_stride = stride if (reduction and edge["input_node"] < self.num_input_nodes) else 1

                op_fn = OPERATIONS.get(op_name, OPERATIONS["zero"])
                node_ops.append(op_fn(channels, edge_stride))
                node_edges.append(edge["input_node"])

            self.nodes.append(node_ops)
            self.edge_info.append(node_edges)

    def forward(self, inputs):
        """
        Forward pass through the cell.
        inputs: list of tensors from previous cells
        """
        states = []
        for i, inp in enumerate(inputs):
            states.append(self.preprocess[i](inp))

        for node_ops, edges in zip(self.nodes, self.edge_info):
            node_output = sum(
                op(states[edge_idx]) for op, edge_idx in zip(node_ops, edges)
            )
            states.append(node_output)

        # Output is the mean of all intermediate node outputs
        intermediate_states = states[self.num_input_nodes:]
        return sum(intermediate_states) / len(intermediate_states)


class NASNetwork(nn.Module):
    """Full network built by stacking cells from NAS architecture."""

    def __init__(self, arch_spec, channels=16, layers=8, num_classes=10):
        super().__init__()
        self.channels = channels
        self.layers = layers

        normal_spec = arch_spec["normal_cell"]
        reduction_spec = arch_spec["reduction_cell"]

        # Stem: initial convolution
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Stack cells
        self.cells = nn.ModuleList()
        reduction_positions = [layers // 3, 2 * layers // 3]

        for i in range(layers):
            if i in reduction_positions:
                cell = Cell(reduction_spec, channels, reduction=True)
            else:
                cell = Cell(normal_spec, channels, reduction=False)
            self.cells.append(cell)

        # Head: global average pool + classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = s0  # Both inputs start as stem output

        for cell in self.cells:
            s_new = cell([s0, s1])
            s0, s1 = s1, s_new

        out = self.global_pool(s1)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


# ─── Training ────────────────────────────────────────────────────────────────

def get_dataset(name, data_path, batch_size):
    """Load dataset with standard augmentation."""
    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, test_loader, 10

    elif name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, test_loader, 100

    else:
        raise ValueError(f"Unsupported dataset: {name}")


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / total, correct / total


def train_architecture(args):
    """Main training function."""
    start_time = time.time()

    # Load architecture
    with open(args.arch, "r") as f:
        arch_spec = json.load(f)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load dataset
    train_loader, test_loader, num_classes = get_dataset(args.dataset, args.data_path, args.batch_size)

    # Build model
    model = NASNetwork(arch_spec, channels=args.channels, layers=args.layers, num_classes=num_classes)
    model = model.to(device)
    params = count_parameters(model)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Learning rate scheduler
    if args.cosine:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)

    # Training loop
    best_val_acc = 0.0
    training_log = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        training_log.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
        })

        # Progress output to stderr
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}",
            file=sys.stderr,
        )

    elapsed = time.time() - start_time

    # Final evaluation
    _, final_acc = evaluate(model, test_loader, criterion, device)

    # Output results as JSON to stdout
    result = {
        "accuracy": round(final_acc, 6),
        "validation_accuracy": round(best_val_acc, 6),
        "parameters": params,
        "training_loss": round(training_log[-1]["train_loss"], 6) if training_log else 0,
        "epochs": args.epochs,
        "training_time_seconds": round(elapsed, 2),
        "device": str(device),
    }

    if args.output_format == "json":
        print(json.dumps(result))
    else:
        print(f"Final Accuracy: {final_acc:.4f}")
        print(f"Best Val Accuracy: {best_val_acc:.4f}")
        print(f"Parameters: {params:,}")
        print(f"Training Time: {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Train NAS architecture")
    parser.add_argument("--arch", required=True, help="Path to architecture JSON file")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--data-path", default="./data", help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--cutout-length", type=int, default=16)
    parser.add_argument("--cosine", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-format", default="json", choices=["json", "text"])
    args = parser.parse_args()

    train_architecture(args)


if __name__ == "__main__":
    main()
