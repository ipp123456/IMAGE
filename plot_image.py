import pickle
import torch
#画图
import matplotlib.pyplot as plt
# device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
with open('record.pkl', 'rb') as f:
    metrics_data = pickle.load(f)

metrics_data = metrics_data

with open('record2.pkl', 'wb') as wt:
    pickle.dump(metrics_data, wt)

plt.figure()
plt.plot(metrics_data['epoch'], metrics_data['train_loss'], 'g-', label='Training loss')
plt.plot(metrics_data['epoch'], metrics_data['val_loss'], 'r-', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.jpg')


plt.figure()
plt.plot(metrics_data['epoch'], metrics_data['train_acc'], 'g-', label='Training acc')
plt.plot(metrics_data['epoch'], metrics_data['val_acc'], 'r-', label='Test acc')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.savefig('acc.jpg')