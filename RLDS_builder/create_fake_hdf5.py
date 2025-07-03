import h5py
import numpy as np
import os

# Output path
os.makedirs("./put_green_pepper_into_pot/train", exist_ok=True)
fake_file_path = "./put_green_pepper_into_pot/train/fake_demo.hdf5"

# Constants
NUM_STEPS = 10
IMAGE_SIZE = 480
ACTION_DIM = 23
STATE_DIM = 23

def generate_fake_image():
    """Generates a random RGB image as a numpy array."""
    return np.random.randint(0, 256, size=(NUM_STEPS, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

with h5py.File(fake_file_path, "w") as f:
    # Main camera image
    f.create_dataset("observation/egocentric_camera", data=generate_fake_image())

    # Left and Right wrist cameras
    f.create_dataset("observation/wrist_image_left", data=generate_fake_image())
    f.create_dataset("observation/wrist_image_right", data=generate_fake_image())

    # Actions and States
    f.create_dataset("actions", data=np.random.randn(NUM_STEPS, ACTION_DIM).astype(np.float32))
    f.create_dataset("state", data=np.random.randn(NUM_STEPS, STATE_DIM).astype(np.float32))

    # Reward
    reward = np.zeros(NUM_STEPS, dtype=np.float32)
    reward[-1] = 1.0  # Reward only on final step
    f.create_dataset("reward", data=reward)

    # Prompt (language instruction)
    dt = h5py.string_dtype(encoding='utf-8')
    f.create_dataset("prompt", data="Put the green pepper into the pot", dtype=dt)

print(f"Fake HDF5 file created at: {fake_file_path}")
