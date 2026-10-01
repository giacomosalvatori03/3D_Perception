import os
import urllib.request

WEIGHTS_DIR = "weights"
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, "pointpillar_kitti.pth")

# URL ufficiale e diretto dal CDN OpenMMLab per PointPillars (KITTI 3D Car/3Class)
WEIGHTS_URL = "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"

def download_pointpillars_weights():
    """Scarica i pesi pre-addestrati di PointPillars dal CDN di OpenMMLab."""
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    if os.path.exists(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > 0:
        print(f"✅ Pesi PointPillars gia presenti in: {WEIGHTS_FILE}")
        return WEIGHTS_FILE

    print(f"⬇️ Scaricamento pesi PointPillars pre-addestrati (~20 MB)...")
    
    try:
        # Aggiungiamo un User-Agent per evitare blocchi HTTP
        req = urllib.request.Request(
            WEIGHTS_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(WEIGHTS_FILE, 'wb') as out_file:
            out_file.write(response.read())
            
        print(f"🎉 Download completato con successo: {WEIGHTS_FILE}")
    except Exception as e:
        print(f"❌ Errore durante il download dei pesi: {e}")
        return None
        
    return WEIGHTS_FILE

if __name__ == "__main__":
    download_pointpillars_weights()