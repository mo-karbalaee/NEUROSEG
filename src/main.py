import tifffile as tiff
import matplotlib.pyplot as plt

def load_and_inspect(path):
    data = tiff.imread(path)

    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min value:", data.min())
    print("Max value:", data.max())

    return data


data = load_and_inspect("../data/6s.tif")

def play(data, step=1):
    plt.figure()

    for t in range(0, data.shape[0], step):
        plt.imshow(data[t], cmap="gray")
        plt.title(f"t = {t}")
        plt.axis("off")

        plt.pause(0.05)
        plt.clf()

    plt.close()


play(data)