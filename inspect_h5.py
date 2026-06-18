import h5py

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape}")

with h5py.File("backend/weights/ddpm_mnist_cond_best.weights.h5", "r") as f:
    f.visititems(print_structure)
