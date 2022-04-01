from datetime import datetime

from fastapi import APIRouter, Depends, File, UploadFile
from starlette.responses import JSONResponse
from starlette.requests import Request

from sqlalchemy.orm import Session
from app.database.schema import SkinDatas

from app.database.conn import db
from os import path
from random import random, randint, choice

IMG_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'skindata')

router = APIRouter()


@router.get("/test")
async def index():
    """
    ELB 상태 체크용 API
    :return:
    """
    current_time = datetime.utcnow()
    #return Response(f"Notification API (UTC: {current_time.strftime('%Y.%m.%d %H:%M:%S')})")
    return JSONResponse(status_code=200, content=dict(noti=f"(UTC: {current_time.strftime('%Y.%m.%d %H:%M:%S')})"))


@router.post("/testimg")
async def post_skin(request: Request, file: UploadFile = File(...), session: Session = Depends(db.session)):
    """
    run Skin
    :param request:
    """
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return JSONResponse(status_code=400, content=dict(msg="image must be jpg or png format"))

    file_name = file.filename
    file_path = path.join(IMG_DIR, file_name)
    print(file_path)
    with open(file_path, "wb+") as file_object:
        file_object.write(file.file.read())
    current_time = datetime.utcnow()
    new_data = SkinDatas.create(session, auto_commit=True, user_id=1, **skin_check())
    return JSONResponse(status_code=200, content=dict(result="succesful", noti=f"(UTC: {current_time.strftime('%Y.%m.%d %H:%M:%S')})", file_name=file_name))


def skin_check():
    wrinkle = round(random(), 2)
    skin_tone = randint(1, 10)
    pore_detect = round(random(), 2)
    dead_skin = round(random(), 2)
    oilly = choice(["oilly", "normal", "dry"])
    pih = randint(1, 10)



    return dict(wrinkle=wrinkle, skin_tone=skin_tone,
        pore_detect=pore_detect, dead_skin=dead_skin, oilly=oilly, pih=pih)