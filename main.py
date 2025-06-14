import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import os


# CNN for CIFAR-10
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # RGB input
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x  # Softmax applied in prediction


# Load CIFAR-10 dataset
def load_cifar10_data():
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        return trainloader, testloader, testset, classes, None
    except Exception as e:
        return None, None, None, None, f"Error loading CIFAR-10 data: {str(e)}"


# Train the model
def train_model(model, trainloader, epochs=5):
    try:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        losses = []
        total_batches = len(trainloader)
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for i, (images, labels) in enumerate(trainloader, 1):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                if i % 100 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {i}/{total_batches}, Loss: {loss.item():.4f}")
            avg_loss = epoch_loss / total_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        return True, losses, None
    except Exception as e:
        return False, None, f"Error training model: {str(e)}"


# Evaluate the model
def evaluate_model(model, testloader):
    try:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy, None
    except Exception as e:
        return 0.0, f"Error evaluating model: {str(e)}"


# Predict for a single image
def predict_image(model, image, classes):
    try:
        model.eval()
        image_tensor = image.unsqueeze(0)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()[0]
            _, predicted = torch.max(outputs, 1)
        return classes[predicted.item()], probabilities, None
    except Exception as e:
        return None, None, f"Error predicting image: {str(e)}"


# Save sample images
def save_sample_images(testset, classes, num_samples=5):
    try:
        plt.figure(figsize=(10, 2))
        indices = np.random.randint(0, len(testset), num_samples)
        for i, idx in enumerate(indices):
            image, label = testset[idx]
            image_np = image.permute(1, 2, 0).numpy() * 0.5 + 0.5  # Denormalize
            image_np = np.clip(image_np, 0, 1)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(image_np)
            plt.title(classes[label])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=100)
        plt.close()
        return True, None
    except Exception as e:
        return False, f"Error saving sample images: {str(e)}"


# Save training loss plot
def save_training_loss(losses):
    try:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(losses) + 1), losses, 'b-', label='Training Loss')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('training_loss.png', dpi=100)
        plt.close()
        return True, None
    except Exception as e:
        return False, f"Error saving training loss: {str(e)}"


# Save prediction plot
def save_prediction_plot(image, label, prediction, probabilities, classes, index):
    try:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        image_np = image.permute(1, 2, 0).numpy() * 0.5 + 0.5
        image_np = np.clip(image_np, 0, 1)
        plt.imshow(image_np)
        plt.title(f"True: {label}\nPred: {prediction}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.bar(classes, probabilities, color='skyblue')
        plt.title('Prediction Confidence')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'prediction_{index}.png', dpi=100)
        plt.close()
        return True, None
    except Exception as e:
        return False, f"Error saving prediction plot: {str(e)}"


# Tkinter GUI
class CIFAR10GUI:
    def __init__(self, root, model, testset, classes, accuracy):
        self.root = root
        self.model = model
        self.testset = testset
        self.classes = classes
        self.root.title("CIFAR-10 Image Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        tk.Label(root, text="CIFAR-10 Image Classifier", font=("Helvetica", 16, "bold"), bg="#f0f0f0").pack(pady=10)
        tk.Label(root, text=f"Model Accuracy: {accuracy:.2%}", font=("Arial", 12), bg="#f0f0f0").pack()

        frame = tk.Frame(root, bg="#f0f0f0")
        frame.pack(pady=5)
        tk.Label(frame, text="Select Test Image Index (0-19):", bg="#f0f0f0").pack(side=tk.LEFT)
        self.index_var = tk.StringVar(value="0")
        indices = [str(i) for i in range(20)]
        ttk.OptionMenu(frame, self.index_var, "0", *indices).pack(side=tk.LEFT, padx=10)

        tk.Button(root, text="Predict", command=self.predict, font=("Arial", 12), bg="#4CAF50", fg="white", padx=10,
                  pady=5).pack(pady=10)

        self.pred_label = tk.Label(root, bg="#ffffff", bd=2, relief="solid")
        self.pred_label.pack(pady=10)

        self.output_text = scrolledtext.ScrolledText(root, height=6, width=50, wrap=tk.WORD, font=("Arial", 10))
        self.output_text.pack(pady=10)
        self.output_text.insert(tk.END,
                                "Select an index and click Predict.\nView training_loss.png and sample_images.png in the project folder.\n")
        self.output_text.config(state='disabled')

    def predict(self):
        try:
            index = int(self.index_var.get())
            image, label = self.testset[index]
            true_label = self.classes[label]
            prediction, probabilities, error = predict_image(self.model, image, self.classes)
            if error:
                messagebox.showerror("Error", error)
                return

            success, error = save_prediction_plot(image, true_label, prediction, probabilities, self.classes, index)
            if not success:
                messagebox.showerror("Error", error)
                return

            if os.path.exists(f'prediction_{index}.png'):
                pred_img = Image.open(f'prediction_{index}.png')
                pred_img = pred_img.resize((400, 200), Image.Resampling.LANCZOS)
                pred_photo = ImageTk.PhotoImage(pred_img)
                self.pred_label.config(image=pred_photo)
                self.pred_label.image = pred_photo
            else:
                messagebox.showerror("Error", f"Prediction image not found: prediction_{index}.png")
                return

            self.output_text.config(state='normal')
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Index: {index}\n")
            self.output_text.insert(tk.END, f"True Label: {true_label}\n")
            self.output_text.insert(tk.END, f"Predicted Label: {prediction}\n")
            self.output_text.insert(tk.END, "Confidence Scores:\n")
            for cls, prob in zip(self.classes, probabilities):
                self.output_text.insert(tk.END, f"  {cls}: {prob:.4f}\n")
            self.output_text.config(state='disabled')
            self.output_text.see(tk.END)

        except ValueError:
            messagebox.showerror("Error", "Invalid index. Select a number from the dropdown.")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {str(e)}")


def main():
    print("Initializing CIFAR-10 Image Classifier...")

    trainloader, testloader, testset, classes, error = load_cifar10_data()
    if trainloader is None:
        print(f"Error: {error}")
        return

    print("Training model...")
    model = CIFAR10_CNN()
    success, losses, error = train_model(model, trainloader)
    if not success:
        print(f"Error: {error}")
        return

    accuracy, error = evaluate_model(model, testloader)
    if error:
        print(f"Error: {error}")
        return
    print(f"Model trained successfully! Test accuracy: {accuracy:.2%}")

    print("Generating training loss plot (saved as training_loss.png)...")
    success, error = save_training_loss(losses)
    if not success:
        print(f"Error: {error}")

    print("Generating sample images plot (saved as sample_images.png)...")
    success, error = save_sample_images(testset, classes)
    if not success:
        print(f"Error: {error}")

    root = tk.Tk()
    app = CIFAR10GUI(root, model, testset, classes, accuracy * 100)
    root.mainloop()


if __name__ == "__main__":
    main()