import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- PARAMETERS ---
OUTPUT_DIR = "synthetic-images-pd"
LABELS_DIR = "labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

CSV_PATH = os.path.join(LABELS_DIR, "z_labels_pd.csv")

# Z distances (micrometers)
Z_LEVELS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Imaging parameters
params = {
    'wavelength': 0.530,    # µm
    'pixel_size': 0.3733,   # µm
    'patch_size': 512,      # px
    'ref_ind': 1.00         # air
}

# --- SIMPLE PHYSICS SIMULATION ---
class AxialPlane:
    def __init__(self, init_guess, z0, params):
        self.z0 = z0
        self.wl = params['wavelength'] / params['ref_ind']
        self.k = 1 / self.wl
        self.n_pixel = params['patch_size']
        self.pixel_size = params['pixel_size']
        self.max_freq = 1 / self.pixel_size
        self.comp_field = init_guess

    def __call__(self, z):
        ang_spectrum = np.fft.fftshift(np.fft.fft2(self.comp_field))
        uu, vv = np.meshgrid(
            self.max_freq * np.arange(-self.n_pixel/2, self.n_pixel/2) / self.n_pixel,
            self.max_freq * np.arange(-self.n_pixel/2, self.n_pixel/2) / self.n_pixel
        )
        mask = ((self.k**2 - uu**2 - vv**2) >= 0).astype('float32')
        ww = np.sqrt((self.k**2 - uu**2 - vv**2) * mask).astype('float32')
        h = np.exp(1j * 2 * np.pi * ww * (z - self.z0)) * mask
        ang_spectrum *= h
        prop_field = np.fft.ifft2(np.fft.ifftshift(ang_spectrum))
        return prop_field.real, prop_field.imag


def random_amplitude(size):
    """Generate random texture as amplitude."""
    return np.random.rand(size, size).astype('float32')

def gedanken_pattern(size=512, physical_period=9, pixel_size=0.37, theta=None, missing_rate=0.3):
    """
    Generate a random 2D periodic dot pattern (Gedanken-style) with proper mask size.
    """
    if theta is None:
        theta = np.random.uniform(0, np.pi)

    y, x = np.meshgrid(np.arange(size), np.arange(size))
    x = x - size / 2
    y = y - size / 2

    # rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # compute period in pixels
    pixel_period = physical_period / pixel_size

    # cosine-based periodic pattern
    phase_x = 2 * np.pi * x_rot / pixel_period
    phase_y = 2 * np.pi * y_rot / pixel_period
    pattern = (0.5 + 0.5 * np.cos(phase_x)) * (0.5 + 0.5 * np.cos(phase_y))

    # random mask for missing dots
    grid_size = int(np.ceil(size / pixel_period))
    mask = np.random.rand(grid_size, grid_size) > missing_rate

    # expand mask to full image
    mask_full = np.kron(mask, np.ones((int(pixel_period), int(pixel_period))))
    # pad if smaller than size
    pad_y = size - mask_full.shape[0]
    pad_x = size - mask_full.shape[1]
    if pad_y > 0 or pad_x > 0:
        mask_full = np.pad(mask_full, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=1)

    mask_full = mask_full[:size, :size]  # ensure exact shape
    pattern *= mask_full
    pattern = pattern / pattern.max()
    return pattern.astype('float32')


def apply_focus(amplitude, z, params):
    """Apply a quadratic phase curvature to simulate defocus."""
    n = params['patch_size']
    pixel = params['pixel_size']
    wl = params['wavelength']
    y, x = np.meshgrid(np.arange(n) - n/2, np.arange(n) - n/2)
    r2 = (x * pixel)**2 + (y * pixel)**2
    phase = np.exp(1j * np.pi * r2 / (wl * z))  # quadratic phase
    return amplitude * phase

def generate_scene_defocus(scene_id, csv_file, counter):
    amp = gedanken_pattern(size=params['patch_size'])

    for z in Z_LEVELS:
        amp_z = apply_focus(amp, z, params)  # add phase curvature
        wave = AxialPlane(amp_z, 0, params)
        re, im = wave(z)
        intensity = np.sqrt(re**2 + im**2).astype('float32')

        z_str = f"{int(z):03d}"
        img_name = f"scene{scene_id:05d}_z{z_str}um.png"
        plt.imsave(os.path.join(OUTPUT_DIR, img_name), intensity, cmap='gray')

        csv_file.write(f"{img_name},{z}\n")
        counter[0] += 1
        if counter[0] % 1000 == 0:
            csv_file.flush()



def main():
    total_images_desired = 100_000
    total_scenes = total_images_desired // len(Z_LEVELS)  # 8333 for 12 Z-levels
    counter = [0]

    with open(CSV_PATH, "w") as csv_file:
        csv_file.write("image_name,z_distance_um\n")  # header

        for scene_id in tqdm(range(total_scenes), desc="Generating synthetic holograms with phase defocus"):
            generate_scene_defocus(scene_id, csv_file, counter)

        csv_file.flush()

    print("✅ Generation complete.")
    print(f"Images saved to: {OUTPUT_DIR}")
    print(f"Labels saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
