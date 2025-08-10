from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from ..models.io import DiagnoseIn
from .diagnose import diagnose
from ..exporters.ppt import build_ppt

router = APIRouter()

@router.post("/export/pptx")
def export_pptx(req: DiagnoseIn):
    data = diagnose(req)
    deck = build_ppt(data, title=f"Benchmark – {req.category} / {req.subcategory}")
    filename = f"benchmark_{req.category}_{req.subcategory}.pptx".replace(" ", "_")
    return StreamingResponse(
        deck,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
