import matplotlib.pyplot as plt

# Sample data for accuracy and F1 score for 10 epochs
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
accuracy_data = [0.6833333333333333, 0.7266666666666667, 0.7266666666666667, 0.7033333333333334, 0.715, 0.7266666666666667, 0.7133333333333334, 0.7266666666666667, 0.7066666666666667, 0.7133333333333334]
f1_score_data = [0.7020908501667007, 0.7312218397512341, 0.7294340109259705, 0.7076129650085642, 0.7143915949836747, 0.7274636315461432, 0.7121707683198577, 0.7223781664627744, 0.7089982289982292, 0.7112632207167203]

# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracy_data, marker='o', color='green', linestyle='-')
for i, txt in enumerate(accuracy_data):
    plt.annotate(f"{txt:.2f}", (epochs[i], accuracy_data[i]))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(epochs)  # Show all epoch numbers
plt.title('Model Accuracy over Epochs')
plt.grid(True)
plt.show()

# Plotting F1 score
plt.figure(figsize=(10, 5))
plt.plot(epochs, f1_score_data, marker='o', color='blue', linestyle='-')
for i, txt in enumerate(f1_score_data):
    plt.annotate(f"{txt:.2f}", (epochs[i], f1_score_data[i]))
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.xticks(epochs)  # Show all epoch numbers
plt.title('Model F1 Score over Epochs')
plt.grid(True)
plt.show()

loss_data = {
    1: [1.0886, 1.0642, 1.0446, 0.9395, 0.9731, 0.7659, 0.9715, 0.6270, 0.6039, 0.8729, 0.6106, 0.3577, 0.5225, 0.6220, 0.6899, 0.5688, 0.4778, 0.4120, 0.4647, 0.6921],
    2: [0.3053, 0.3251, 0.2825, 0.2434, 0.2543, 0.3434, 0.5175, 0.2141, 0.3044, 0.3667, 0.3465, 0.3341, 0.1077, 0.0509, 0.3468, 0.1206, 0.3144, 0.3937, 0.1105, 0.2559],
    3: [0.0922, 0.1124, 0.1545, 0.1529, 0.0481, 0.1104, 0.2030, 0.1223, 0.0749, 0.0276, 0.0212, 0.0745, 0.1838, 0.0480, 0.1065, 0.1217, 0.0626, 0.0323, 0.0205, 0.1140],
    4: [0.2134, 0.1389, 0.0719, 0.0372, 0.0106, 0.0096, 0.0197, 0.0680, 0.0114, 0.0279, 0.0062, 0.1421, 0.0500, 0.1352, 0.0378, 0.1015, 0.3149, 0.0443, 0.1481, 0.0642],
    5: [0.0076, 0.0138, 0.0096, 0.0118, 0.0320, 0.0084, 0.0106, 0.0110, 0.0075, 0.0048, 0.0049, 0.1653, 0.0073, 0.0330, 0.0970, 0.0067, 0.0077, 0.0105, 0.0281, 0.0437],
    6: [0.1517, 0.0831, 0.0136, 0.0057, 0.0049, 0.0038, 0.1938, 0.0186, 0.1435, 0.0176, 0.0355, 0.2064, 0.0283, 0.0116, 0.0065, 0.0057, 0.1069, 0.0942, 0.0644, 0.0510],
    7: [0.0110, 0.0421, 0.0024, 0.0184, 0.0240, 0.0057, 0.0045, 0.0082, 0.0987, 0.0620, 0.0255, 0.0073, 0.0441, 0.1425, 0.0625, 0.0099, 0.0073, 0.1609, 0.0107, 0.0464],
    8: [0.0633, 0.1121, 0.0146, 0.0035, 0.1107, 0.0071, 0.0051, 0.0135, 0.0024, 0.0125, 0.0019, 0.0040, 0.0017, 0.0026, 0.0045, 0.0173, 0.0195, 0.0049, 0.0016, 0.0308],
    9: [0.0035, 0.0723, 0.0102, 0.0040, 0.0018, 0.0020, 0.0040, 0.1002, 0.0141, 0.0199, 0.0096, 0.0461, 0.0247, 0.0055, 0.0974, 0.0085, 0.0031, 0.0074, 0.0031, 0.0334],
    10: [0.0162, 0.0032, 0.0024, 0.0024, 0.0019, 0.0130, 0.0012, 0.0370, 0.0013, 0.0016, 0.0203, 0.0040, 0.0129, 0.0019, 0.0007, 0.0514, 0.0102, 0.0081, 0.0031, 0.0200]
}

# Flatten the loss data for plotting
flat_loss_data = []
for epoch_losses in loss_data.values():
    flat_loss_data.extend(epoch_losses)

# Generate batch indices for x-axis
batch_indices = list(range(1, len(flat_loss_data) + 1))

# Plotting loss
plt.figure(figsize=(15, 5))
plt.plot(batch_indices, flat_loss_data, marker='o', color='red', linestyle='-')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Model Loss per Batch over Epochs')
plt.grid(True)
plt.savefig('loss_plot.png')
plt.show()