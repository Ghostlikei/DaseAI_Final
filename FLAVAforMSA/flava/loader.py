# imports
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import re

from transformers import AutoProcessor

# Dataset loader
class MSADataset(Dataset):
    def __init__(self, data_folder, number_list, transform=None, resample=True, label_file='train.txt'):
        self.data_folder = data_folder
        self.number_list = number_list
        self.transform = transform
        self.processor = AutoProcessor.from_pretrained("facebook/flava-full")

        # Read labels
        self.labels = {}
        with open(os.path.join(data_folder, label_file), 'r') as file:
            for line in file:
                number, label = line.strip().split(',')
                self.labels[number] = {'positive': 0, 'negative': 1, 'neutral': 2, 'null': 404}[label]

        if resample:
            self.resample()

    def __len__(self):
        return len(self.number_list)
    
    def resample(self):
        # Count labels
        label_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        for number in self.number_list:
            label = self.labels[number]
            if label == 0:
                label_counts['positive'] += 1
            elif label == 1:
                label_counts['negative'] += 1
            else:
                label_counts['neutral'] += 1

        # Find maximum count to balance
        max_count = max(label_counts.values())

        # Resample
        new_number_list = []
        for label, count in label_counts.items():
            label_numbers = [num for num in self.number_list if self.labels[num] == {'positive': 0, 'negative': 1, 'neutral': 2}[label]]
            new_number_list.extend(random.choices(label_numbers, k=max_count))

        random.shuffle(new_number_list)
        self.number_list = new_number_list

    def __getitem__(self, idx):
        MAX_LEN = 75

        number = self.number_list[idx]

        # Load image
        image_path = os.path.join(self.data_folder, 'data', f'{number}.jpg')
        image = Image.open(image_path).convert('RGB')

        # Load text
        text_path = os.path.join(self.data_folder, 'data', f'{number}.txt')
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()[:MAX_LEN]
        except UnicodeDecodeError:
            with open(text_path, 'r', encoding='utf-8', errors='replace') as file:
                text = file.read().strip()[:MAX_LEN]
        text = re.sub(r'[^a-zA-Z]', ' ', text).lower()
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        return inputs, self.labels[number]

def custom_collate_fn(batch):
    # Initialize empty lists to hold the elements of each input and the labels
    input_ids, attention_masks, image_tensors, labels = [], [], [], []
    
    for item, label in batch:
        # Process the outputs from the processor
        input_ids.append(item['input_ids'].squeeze(0))  # Remove batch dimension
        attention_masks.append(item['attention_mask'].squeeze(0))
        image_tensors.append(item['pixel_values'].squeeze(0))
        labels.append(label)

    # Pad the sequences to the longest one in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Stack the image tensors and labels
    image_tensors = torch.stack(image_tensors)
    labels = torch.tensor(labels)

    # Return a dictionary for the inputs and a tensor for the labels
    return {'input_ids': input_ids, 'attention_mask': attention_masks, 'pixel_values': image_tensors}, labels


# Dataloader of Flava-Processor
def get_data_loaders(data_folder, batch_size=32, val_split=0.15, label_file='train.txt'):
    # Load and split number list
    with open(os.path.join(data_folder, 'train.txt'), 'r') as file:
        number_list = [line.split(',')[0] for line in file]
    random.shuffle(number_list)
    split = int(len(number_list) * val_split)
    train_numbers, val_numbers = number_list[split:], number_list[:split]

    # Create datasets
    train_dataset = MSADataset(data_folder, train_numbers, None, label_file=label_file)
    val_dataset = MSADataset(data_folder, val_numbers, None, label_file=label_file, resample=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_data_loaders('../dataset', batch_size=32)

    print(train_loader)
    # Peek the first five data in the dataloader
    flag = 0
    for each_input, labels in train_loader:
        if flag == 0: 
            print(each_input)
            flag = 1
