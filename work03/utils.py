import matplotlib.pyplot as plt
import numpy as np

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# data visualization
def show_one_data(data, label):
    # data: (3, 32, 32) torch.Tensor
    # labels: torch.Tensor
    tensor = data
    img = np.transpose(tensor.numpy().astype(int), (1, 2, 0))
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title("label {} : {}".format(label.item(),class_names[label]))
    plt.show()


def show_one_np_data(data):
    # data: numpy.ndarray
    if data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))
    img = data.astype(int)
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.show()