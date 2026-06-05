#!/usr/bin/env python3
"""Quick test to verify hybrid_audit works with minimal items."""

import os

os.environ["N_ITEMS"] = "10"  # Very small test
os.environ["BATCH_SIZE"] = "5"
os.environ["SHOW_TRAJECTORY"] = "1"  # Show trajectory for debugging

# Now run the main
import examples.hybrid_audit

examples.hybrid_audit.main()
