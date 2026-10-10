"""The JS contract tables, generated into _contract.json by
scripts/gen-backtest-contract.mjs. Never edited by hand."""

import json
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_contract.json"), encoding="utf-8") as _f:
    CONTRACT = json.load(_f)
