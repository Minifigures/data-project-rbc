"""Root entry point for Streamlit Community Cloud (free hosting).

Community Cloud looks for `streamlit_app.py` at the repo root and installs
`requirements.txt`. Because the package uses a src layout, we add `src` to the
path before importing the dashboard, so `import claimguard` resolves without an
editable install.

Locally you can still run the dashboard directly with:
    streamlit run src/claimguard/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from claimguard.dashboard.app import main  # noqa: E402

main()
