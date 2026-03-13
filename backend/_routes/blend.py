"""Route handler for POST /api/blend."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import BlendRequest, BlendResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["blend"])


@router.post("/blend", response_model=BlendResponse)
def route_blend(req: BlendRequest, handler: AppHandler = Depends(get_state_service)) -> BlendResponse:
    return handler.blend.run(req)
