"""ESMA-MILAN HTTP API.

FastAPI service wrapping the ESMA -> MILAN pipeline. See `server.py`
for the app and endpoints, `handlers.py` for the request-handling
logic, and `schemas.py` for the response models.
"""

from esma_milan.api.server import app

__all__ = ["app"]
