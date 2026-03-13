"""Route handler for /api/loras."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import LoraListResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api", tags=["loras"])


@router.get("/loras", response_model=LoraListResponse)
def route_list_loras(
    handler: AppHandler = Depends(get_state_service),
) -> LoraListResponse:
    """GET /api/loras — list available LoRA files."""
    return handler.video_generation.list_loras()
