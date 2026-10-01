import os
import urllib.request

WEIGHTS_DIR = "weights"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "pointpillar_kitti.pth")

# URL del checkpoint PointPillars pre-addestrato su KITTI
WEIGHTS_URL = "https://github.com/open-mmlab/mmdetection3d/releases/download/v0.6.0_models/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20200620_230614-70e1b7d7.pth"

def download_pointpillars_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if os.path.exists(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 0:
        print(f"✅ Pesi PointPillars già presenti in: {WEIGHTS_FILE}")
        return WEIGHTS_FILE

    print(f"⬇️ Scaricamento pesi PointPillars pre-addestrati (~20 MB)...")
    urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE)
    print(f"🎉 Download completato con successo: {WEIGHTS_FILE}")
    return WEIGHTS_FILE

if __name__ == "__main__":
    download_pointpillars_weights()