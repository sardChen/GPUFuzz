"""Per-target assembly emission (dialect + macro contract)."""
from .emit import AssemblyEmitter, EmitResult, emit_program, load_profile, write_assembly

__all__ = ["AssemblyEmitter", "EmitResult", "emit_program", "load_profile", "write_assembly"]
