from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gestures.features import FEATURE_DIMENSION, FEATURE_SCHEMA


def main() -> None:
    contract = FEATURE_SCHEMA.as_dict()
    contract["dimension"] = FEATURE_DIMENSION
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
