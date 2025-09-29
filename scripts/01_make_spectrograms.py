# scripts/01_make_spectrograms.py
import os, numpy as np, cv2
from pathlib import Path
from tqdm import tqdm
from scipy.signal import stft
import scipy.io as sio

SR = 20_000_000   # DroneRF recorded at 20 MHz sample rate
NFFT = 1024
HOP = 256

def load_iq(path):
    """
    Load DroneRF .mat files.
    Each file usually contains a dict with key 'rx_signal' or 'data'.
    Adjust the key if needed by inspecting one file.
    """
    mat = sio.loadmat(path)
    # Typical structure: {'__header__':..., '__version__':..., '__globals__':..., 'data': array([...])}
    if 'data' in mat:
        iq = mat['data'].squeeze()
    elif 'rx_signal' in mat:
        iq = mat['rx_signal'].squeeze()
    else:
        raise ValueError(f"Unknown keys in {path}: {mat.keys()}")
    # Convert to complex if stored as 2 columns [I, Q]
    if np.isrealobj(iq) and iq.ndim == 2 and iq.shape[1] == 2:
        iq = iq[:,0] + 1j*iq[:,1]
    return iq.astype(np.complex64)

def iq_to_spectrogram(iq: np.ndarray):
    f, t, Z = stft(iq, fs=SR, nperseg=NFFT, noverlap=NFFT-HOP, return_onesided=False)
    S = np.abs(Z)
    S = 20*np.log10(S + 1e-6)
    # normalize 0–255
    S = (S - S.min()) / (S.max() - S.min() + 1e-9)
    img = (S*255).astype(np.uint8)
    return img

def save_spec(img, out_path):
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), img)

def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = list(Path(in_dir).glob("**/*.mat"))
    print(f"Found {len(files)} files")
    for fpath in tqdm(files):
        try:
            iq = load_iq(fpath)
            img = iq_to_spectrogram(iq)
            save_spec(img, Path(out_dir) / (fpath.stem + ".png"))
        except Exception as e:
            print(f"Error on {fpath}: {e}")

if __name__ == "__main__":
    main("data_raw/dronerf", "data_work/spectrograms")
