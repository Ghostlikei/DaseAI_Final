import torch
import os
from torch import nn, optim
from transformers import BertTokenizer
from sklearn.metrics import f1_score
import numpy as np

from .flava_model import FlavaForClassification
from .loader import get_data_loaders, MSADataset, DataLoader, custom_collate_fn

class ModelRunner:
    def __init__(self, dataset_path='../dataset', lr = 0.0001, batch_size = 32, dropout = 0.3, class_weights=None, mask_text = False, mask_image = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlavaForClassification(dropout=dropout).to(self.device)
        self.batch_size = batch_size

        # Applying weighted cross entropy loss due to the inbalanced labels
        if class_weights is None:
            class_weights = torch.tensor([1.0, 1.0, 1.0])  # Default to equal weights
        else:
            class_weights = torch.tensor(class_weights)

        self.mask_text = mask_text
        self.mask_image = mask_image

        self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader, self.val_loader = get_data_loaders(dataset_path, batch_size=batch_size)

    def train(self, epoch = -1):
        # Train for one epoch
        self.model.train()
        total_loss = 0

        for cnt, (inputs, labels) in enumerate(self.train_loader):
            if self.mask_image:
                # mask image here
                inputs['pixel_values'] = torch.zeros_like(inputs['pixel_values'])
            if self.mask_text:
                # mask text here
                inputs['input_ids'] = torch.zeros_like(inputs['input_ids'])
                inputs['attention_mask'] = torch.zeros_like(inputs['attention_mask'])
            labels = labels.to(self.device)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            
            self.optimizer.step()
            total_loss += loss.item()
            if cnt % 10 == 0:
                print(f"Epoch {epoch}, Batch {cnt + 1}/{len(self.train_loader)}, loss = {loss.item()}")
        
        print(f"Training Epoch {epoch} ended, loss = {total_loss / cnt}")

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            all_labels = []
            all_predictions = []

            for inputs, labels in self.val_loader:
                if self.mask_image:
                    # mask image here
                    inputs['pixel_values'] = torch.zeros_like(inputs['pixel_values'])
                if self.mask_text:
                    # mask text here
                    inputs['input_ids'] = torch.zeros_like(inputs['input_ids'])
                    inputs['attention_mask'] = torch.zeros_like(inputs['attention_mask'])
                labels = labels.to(self.device)
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                outputs = self.model(inputs)
                # print("Logits: ", outputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # The logits are raw scores. To get class probabilities, apply softmax
                probabilities = torch.softmax(outputs, dim=1)

                # For actual predictions, take the argmax
                predicted = torch.argmax(probabilities, dim=1)

                # print("Predicted: ", predicted)
                # print("Label: ", labels)
                total_correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

            accuracy = total_correct / len(self.val_loader.dataset)
            f1 = f1_score(all_labels, all_predictions, average='weighted')

            print(f"Validation Loss: {total_loss / len(self.val_loader)}, Accuracy: {accuracy}, F1 Score: {f1}")

    def predict(self, dataset_path, output_file):
        # Load predict data
        predict_loader, number_list = self.get_predict_loader(dataset_path)

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs, numbers in predict_loader:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(probabilities, dim=1)
                predictions.extend(zip(numbers, predicted.cpu().numpy()))

        # Map numerical labels to string labels
        label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

        # Write predictions to output file
        with open(output_file, 'w') as f:
            for i, (number, label) in enumerate(predictions):
                label_string = label_map[label]
                f.write(f"{number_list[i]},{label_string}\n")

    def get_predict_loader(self, dataset_path):
        # Read predict.txt
        with open(os.path.join(dataset_path, 'test_without_label.txt'), 'r') as file:
            number_list = [line.split(',')[0] for line in file]

        # Create predict dataset
        predict_dataset = MSADataset(dataset_path, number_list, label_file='test_without_label.txt', transform=None, resample=False)

        # Create predict loader
        predict_loader = DataLoader(predict_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)

        return predict_loader, number_list

if __name__ == '__main__':
    current_seed = torch.initial_seed()
    print("Current PyTorch seed:", current_seed)
    # Testing correctness here
    epoch = 5

    runner = ModelRunner(batch_size=32)

    for i in range(epoch):
        runner.train(i+1)
        runner.validate()