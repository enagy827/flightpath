"""Load factor is produced by the demand module (modules/demand.py): LF = RPM / ASM (proposal eq. 1).
Kept as a separate file for the architecture diagram; no separate estimation."""
from .demand import DemandModule as LoadFactorModule  # noqa: F401
