"""Execution trace adapters: raw DUT JSONL -> canonical Trace."""
from .adapter import SignalExpr, TraceAdapter, load_adapter

__all__ = ["SignalExpr", "TraceAdapter", "load_adapter"]
