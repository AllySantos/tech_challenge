from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("")
async def predict():
    return {}
