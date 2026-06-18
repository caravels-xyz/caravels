"""One-shot competition registration runner.

Run once before June 22:
  uv run python scripts/register.py

Calls `twak compete register` and verifies the result against the competition contract.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from caravels.config import AppConfig
from caravels.twak import TWAKAdapter


def main():
    cfg = AppConfig.from_env()
    twak = TWAKAdapter(stub=not cfg.twak_access_id)

    print("Caravels — Competition Registration")
    print(f"  network:  {cfg.network}")
    print("  contract: 0x212c61b9b72c95d95bf29cf032f5e5635629aed5")
    print(f"  wallet:   {cfg.wallet_address or '(read from TWAK)'}")
    print()

    if not cfg.twak_access_id:
        print("ERROR: TWAK_ACCESS_ID not set. Configure .env first.")
        sys.exit(1)

    status = twak.compete_register()
    print(f"  registration status: {status.value}")

    if status.value == "registered":
        print("  ✓ Registration complete. Submit your agent address on DoraHacks.")
    else:
        print("  ✗ Registration may have failed. Check TWAK output.")
        sys.exit(1)


if __name__ == "__main__":
    main()
