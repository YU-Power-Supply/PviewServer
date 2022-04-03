from os import path
from fastapi import APIRouter, Depends, File, UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database.schema import SkinDatas
from app.models import MySkin
from app.database.conn import db
from app.pview_core import Oilly, PIH, Pore, SkinTone, Wrinkle, recommand

from random import random
from datetime import datetime
import secrets

IMG_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'skindata')

router = APIRouter(prefix="/pview")


@router.post("/skin", status_code=201, response_model=MySkin)
async def post_skin(request: Request, file: UploadFile = File(...), session: Session = Depends(db.session)):
    """
    run Skin
    :param request:
    """
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return JSONResponse(status_code=400, content=dict(msg="image must be jpg or png format"))
    user = request.state.user

    currentTime = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = ''.join([currentTime, secrets.token_hex(16), ".", file.filename.split(".")[1]])
    file_location = path.join(IMG_DIR, file_name)
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    
    skindata, reco = skin_check(file_location)
    SkinDatas.create(session, auto_commit=True, user_id=user.id, file_name=file_name,
                                 **dict(skindata, **reco))
    return dict(id=user.id, skindata=skindata, recomand=reco)


def skin_check(file_location):
    wrinkle = Wrinkle.wrinkleDetect(file_location)
    skin_tone = SkinTone.skinToneDetect(file_location)
    pore_detect = Pore.poreDetect(file_location)
    dead_skin = round(random(), 2)
    oilly = Oilly.oilly(file_location)
    pih = PIH.PIH_model(file_location)

    skindata = dict(wrinkle=wrinkle, skin_tone=skin_tone,
                    pore_detect=pore_detect, dead_skin=dead_skin, oilly=oilly, pih=pih)
    reco = recommand.recommand(**skindata)
    return skindata, reco
