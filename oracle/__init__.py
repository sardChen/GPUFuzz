"""Consistency oracle: runtime status collection, functional and interaction
consistency detectors, bug reproduction and reporting."""
from .oracle import ConsistencyOracle, OracleResult
from .report import BugReport, Reporter
from .reproduce import BugReproducer, ReproResult
from .violation import Violation, dedup

__all__ = ["ConsistencyOracle", "OracleResult", "BugReport", "Reporter",
           "BugReproducer", "ReproResult", "Violation", "dedup"]
