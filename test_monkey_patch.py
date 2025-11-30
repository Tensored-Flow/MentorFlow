# Quick test of monkey patch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import app to trigger monkey patch
try:
    import app
    print("✅ Monkey patch loaded successfully")
    print("✅ App imports OK")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

