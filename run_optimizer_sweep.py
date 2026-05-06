#!/usr/bin/env python3
"""Clear entrypoint for the clean optimizer sweep engine.

The implementation lives in ``run_top_aware_muon_sweep.py`` for backward
compatibility with existing result catalogs and scripts. New code and docs
should invoke this file instead.
"""

from run_top_aware_muon_sweep import main


if __name__ == "__main__":
    main()
