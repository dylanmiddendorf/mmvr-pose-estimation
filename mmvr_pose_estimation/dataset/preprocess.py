import glob

import numpy as np
import numpy.typing as npt

KP_HEIGHT = 480
KP_WIDTH = 640


def load_keypoints(
    keypoint_frame: str,
    *,
    out_height: int = 48,
    out_width: int = 64,
    sigma: int = 2,
) -> npt.NDArray:
    keypoints = np.load(keypoint_frame)["kp"]
    assert keypoints.shape == (1, 17, 3)

    heatmaps = np.zeros((17, out_height, out_width), dtype=np.float16)
    for i, (x, y, confidence) in enumerate(keypoints.squeeze()):
        x *= out_width / KP_WIDTH
        y *= out_height / KP_HEIGHT
        xx, yy = np.meshgrid(np.arange(out_width), np.arange(out_height))
        kp_heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
        kp_heatmap[kp_heatmap < 0.01] = 0.0  # zero out tails
        heatmaps[i] = kp_heatmap

    return heatmaps, keypoints


def load_radar(radar_frame: str) -> tuple[npt.NDArray, npt.NDArray]:
    data = np.load(radar_frame)

    radar_hori = np.log(data["hm_hori"])
    radar_vert = np.log(data["hm_vert"])

    mask_hori = ~np.isnan(radar_hori)
    radar_hori = np.nan_to_num(radar_hori)
    mask_vert = ~np.isnan(radar_vert)
    radar_vert = np.nan_to_num(radar_vert)

    return np.array((radar_hori, mask_hori, radar_vert, mask_vert), dtype=np.float16)


def preprocess_samples(radar_frames: list[str]):
    n = len(radar_frames)

    X = np.zeros((n, 4, 256, 128), dtype=np.float16)  # (n, c, h, w)
    y = np.zeros((n, 17, 48, 64), dtype=np.float16)  # (n, c, h, w)
    kp = np.zeros((n, 17, 3), dtype=np.float32)
    for i, radar_frame in enumerate(radar_frames):
        if i % 900 == 0:
            print(f"Processing {radar_frame}")
        X[i] = load_radar(radar_frame)
        y[i], kp[i] = load_keypoints(radar_frame.replace("_radar", "_pose"))

    np.savez_compressed("data/d1s2.npz", X=X, y=y, kp=kp)


if __name__ == "__main__":
    d1s1_rf = glob.glob("data/raw/d1s1/**/*_radar.npz")
    d1s2_rf = glob.glob("data/raw/d1s2/**/*_radar.npz")
    d3s2_rf = glob.glob("data/raw/d3s2/**/*_radar.npz")
    d4s1_rd = glob.glob("data/raw/d4s1/**/*_radar.npz")
    preprocess_samples(sorted(d1s2_rf))
