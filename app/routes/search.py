from os import path
from fastapi import APIRouter, Depends, File, UploadFile, Path
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from io import BytesIO
import cv2

from app.database.schema import Cosmetics
from app.models import cosmetic
from app.database.conn import db




router = APIRouter(prefix="/search")

@router.post("/goods", status_code = 201) # 화장품 추가하기
async def create_goods(cosmetic: cosmetic, session: Session = Depends(db.session)):
    
    nm_is_exist = await is_goods_nm_exist(cosmetic.goods_nm)
    no_is_exist = await is_goods_no_exist(cosmetic.goods_nm)
    if not cosmetic.goods_nm:
        return JSONResponse(status_code=400, content=dict(msg="goods must be provided"))
    if nm_is_exist or no_is_exist:
        return JSONResponse(status_code=400, content=dict(msg="GOODS_NAME_OR_NUMBER_EXISTS"))
    Cosmetics.create(session, auto_commit=True, goods_no=cosmetic.goods_no, goods_nm=cosmetic.goods_nm,
                     brand_nm=cosmetic.brand_nm,price=cosmetic.price, ingredient=cosmetic.ingredient)
    return JSONResponse(status_code=201, content=dict(msg="GOODS_SUCCESS"))
    

@router.get("/goods/{keyword}") # 화장품 검색
async def read_goods(keyword: str, sort: str = "goods_nm",limit: int = 10):
    
    goods = Cosmetics.filter(goods_nm__like=keyword).order_by(sort).all(limit=limit)
    print("\n\n",goods, "\n\n")
    if not goods:
        return JSONResponse(status_code=400, content=dict(msg="GOODS_NOT_EXISTS"))
    goodsdict = {"length" : len(goods), "cosmeticlist" : []}
    for i, g in enumerate(goods):
        goodsdict["cosmeticlist"].append({"goods_nm" : g.goods_nm, "brand_nm" : g.brand_nm, "category" : g.category, "price" : g.price, "ingredient" : g.ingredient})

    return goodsdict



@router.get("/goods/image/{image_id}") # 썸네일 가져오기~!
async def read_image(image_id: str = Path(...)):
    img = cv2.imread(f"app/skindata/{image_id}.png")
    img_bytes = cv2.imencode('.png', img)[1].tobytes()
    return StreamingResponse(BytesIO(img_bytes), media_type="image/png")


async def is_goods_nm_exist(goods_nm: str):

    get_goods = Cosmetics.get(goods_nm=goods_nm)
    if get_goods:
        return True
    return False

async def is_goods_no_exist(goods_no: str):

    get_goods = Cosmetics.get(goods_no=goods_no)
    if get_goods:
        return True
    return False

