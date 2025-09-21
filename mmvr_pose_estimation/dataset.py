import glob
import os

import numpy as np

KP_HEIGHT = 480
KP_WIDTH = 640


def load_keypoints(session_path: str, index: str) -> np.ndarray:
    # Zero pad the index: 42 -> 00042
    if len(index) < 5:
        index = "0" * (5 - len(index)) + index

    data = np.load(os.path.join(session_path, f"{index}_pose.npz"))
    return data["kp"]


def load_radar(session_path: str, index: str) -> tuple[np.ndarray, np.ndarray]:
    # Zero pad the index: 42 -> 00042
    if len(index) < 5:
        index = "0" * (5 - len(index)) + index

    data = np.load(os.path.join(session_path, f"{index}_radar.npz"))
    hm_hori = np.log(data["hm_hori"])
    hm_vert = np.log(data["hm_vert"])

    return np.asarray([hm_hori, hm_vert])


def get_keypoint_heatmaps(
    keypoints: np.ndarray,
    *,
    out_height: int = 48,
    out_width: int = 64,
    sigma: int = 2,
):
    assert keypoints.shape == (1, 17, 3)

    heatmaps = np.zeros((17, out_height, out_width), dtype=np.float32)
    for i, (x, y, conf) in enumerate(keypoints.squeeze()):
        x *= out_width / KP_WIDTH
        y *= out_height / KP_HEIGHT
        xx, yy = np.meshgrid(np.arange(out_width), np.arange(out_height))
        kp_heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        kp_heatmap[kp_heatmap < 0.01] = 0.0  # zero out tails
        heatmaps[i] = kp_heatmap

    return heatmaps


def preprocess_samples(session_path: str):
    n = len(glob.glob(os.path.join(session_path, "*_pose.npz")))
    X = np.zeros((n, 2, 256, 128), dtype=np.float32)  # (n, c, h, w)
    y = np.zeros((n, 17, 48, 64), dtype=np.float32)  # (n, c, h, w)
    for i in range(n):
        X[i] = load_radar(session_path, str(i))
        kp = load_keypoints(session_path, str(i))
        y[i] = get_keypoint_heatmaps(kp)

    np.savez_compressed("data/processed/d1s1_000.npz", X=X, y=y)


if __name__ == "__main__":
    preprocess_samples("data/raw/P1/d1s1/000")
