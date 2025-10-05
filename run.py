# run.py — jalankan Streamlit tanpa tergantung folder aktif / PowerShell activate
import sys
from pathlib import Path
from streamlit.web import cli as stcli

if __name__ == "__main__":
    app_path = str((Path(__file__).parent / "app.py").resolve())
    # Ganti port kalau perlu (hindari "Port is already in use")
    sys.argv = ["streamlit", "run", app_path, "--server.port", "8503"]
    raise SystemExit(stcli.main())
