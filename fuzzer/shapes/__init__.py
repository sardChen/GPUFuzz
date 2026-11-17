"""Shape catalog and per-shape test case generators."""
from . import catalog, templates
from .genctx import GenCtx
from .spec import RoleSpec, ShapeSpec

__all__ = ["catalog", "templates", "GenCtx", "RoleSpec", "ShapeSpec"]
