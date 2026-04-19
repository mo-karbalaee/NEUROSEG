import tifffile as tiff

def load_and_inspect(path):
    data = tiff.imread(path)

    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min value:", data.min())
    print("Max value:", data.max())

    return data


data = load_and_inspect("../data/6s.tif")