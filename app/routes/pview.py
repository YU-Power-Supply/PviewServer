from os import path
from fastapi import APIRouter, Depends, File, UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse
from sqlalchemy.orm import Session
import cv2
import shutil
import numpy as np
import faiss
import pandas as pd

from app.database.schema import SkinDatas, Recommandation, Cosmetics
from app.models import skin, MySkin
from app.database.conn import db
from app.pview_core import Oilly, PIH, Pore, SkinTone, Wrinkle, recommand, DeadSkin
from app.pview_core.skin_cropper import skin_cropper


from datetime import datetime
import secrets

IMG_DIR = path.join(path.dirname(path.dirname(path.abspath(__file__))), 'skindata')
router = APIRouter(prefix="/pview")

user_vectors_df = pd.read_excel('app/pview_core/userreco.xlsx', index_col=0)
user_vectors_search = user_vectors_df.reset_index()
user_vectors_df = user_vectors_df.set_index('mbrNo')
user_vectors = user_vectors_df.to_numpy(dtype=np.float32)

# Set up Faiss index
d = 6  # dimensionality of the vectors
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index on the user vectors
index.train(user_vectors)

# Add the user vectors to the index
index.add(user_vectors)


@router.post("/globalskin", status_code=201, response_model=skin)
async def post_skin(request: Request, file: UploadFile = File(...), session: Session = Depends(db.session)):
    """
    {run Skin}
    :param request:
    """
    extension = file.filename.split(".")[-1]
    if not extension in ("jpg", "jpeg", "png"):
        return JSONResponse(status_code=400, content=dict(msg="image must be jpg or png format"))
    user = request.state.user

    file_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}{secrets.token_hex(16)}.{extension}'
    file_location = path.join(IMG_DIR, file_name)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.resize(cv2.imread(file_location, cv2.IMREAD_COLOR), (540, 720))
    try :
        _, masked_img = skin_cropper(img, 'all')
    except:
        return JSONResponse(status_code=400, content=dict(msg="it is not a face image."))

    masked_img_256 = cv2.resize(masked_img, (256, 256))

    skindict = {"skin_tone" : SkinTone.skinToneDetect(masked_img_256),
                "pore_detect" : Pore.poreDetect(masked_img_256),
                "pih" : PIH.detect_pih(masked_img_256, "app/pview_core/weights/pih_model/pih_model_weight_0907.h5" )}
    
    return input_skindata(session, file_name, user.id, skindict)


@router.post("/detailskin", status_code=201, response_model=skin)
async def post_skin(request: Request, file: UploadFile = File(...), session: Session = Depends(db.session)):
    """
    {run Skin}
    :param request:
    """
    extension = file.filename.split(".")[-1] 
    if not extension in ("jpg", "jpeg", "png"):
        return JSONResponse(status_code=400, content=dict(msg="image must be jpg or png format"))
    user = request.state.user

    file_name = f'{datetime.now().strftime("%Y%m%d%H%M%S")}{secrets.token_hex(16)}.{extension}'
    file_location = path.join(IMG_DIR, file_name)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = cv2.imread(file_location, cv2.IMREAD_COLOR)


    skindict = {"wrinkle" : Wrinkle.wrinkleDetect(img),
                "dead_skin" : DeadSkin.detect_deadskin(img, "")}
    
    return input_skindata(session, file_name, user.id, skindict)



@router.get("/recommand") # 화장품 추천받기
async def cosmetic_recommand(request: Request):
    
    user = request.state.user
    s_data = SkinDatas.get(user_id=user.id)

    if not s_data:
        return JSONResponse(status_code=400, content=dict(msg="NO_SKINDATA!!"))
    
    s_vector = np.array([s_data.oilly, s_data.dead_skin, s_data.pih, s_data.wrinkle, 
                         s_data.pore_detect, s_data.skin_tone], dtype=np.float32)
    
    np.nan_to_num(s_vector)
    distances, indices = index.search(np.array([s_vector]), 3)
    
    recolist = []
    recodict = {"length" : 3, "cosmeticlist" : []}
    
    for i, idx in enumerate(indices[0]):
        mbr = user_vectors_search.iloc[idx, 1]
        dist = 1.0 / (1.0 + distances[0][i])
        temp = Recommandation.get(mbr_no=mbr)
        recolist += [(temp.recogoods1, float(temp.cossim1)*float(dist)), (temp.recogoods2, float(temp.cossim2)*float(dist)), (temp.recogoods3, float(temp.cossim3)*float(dist))]
    
    recolist.sort(key=lambda x:x[1], reverse=True)
    for reco in recolist[:3]:
        print(reco[0], "\n\n")
        temp = Cosmetics.get(goods_no=reco[0])
        recodict["cosmeticlist"].append({"goods_nm" : temp.goods_nm, "brand_nm" : temp.brand_nm, "category" : temp.category, "price" : temp.price, "ingredient" : temp.ingredient})

    return recodict


@router.post("/pushskindata", status_code=201, response_model=skin) # 강제로 피부데이터 삽입
async def push_skindata(request: Request, skin_list: skin, session: Session = Depends(db.session)):
    
    user = request.state.user
    skindict = {"wrinkle" : skin_list.wrinkle,
                "skin_tone" : skin_list.skin_tone,
                "pore_detect" : skin_list.pore_detect,
                "dead_skin" : skin_list.dead_skin,
                "oilly" : skin_list.oilly,
                "pih" : skin_list.pih
                }
    
    return input_skindata(session, "None", user.id, skindict)
    
    
    
def input_skindata(session : Session, file_name: str, u_id: str, skindict: dict):

    s_data = SkinDatas.get(user_id=u_id)
    if s_data:
        temp = SkinDatas.update(session, auto_commit=True, file_name=file_name,
                                 **skindict)
    else:    
        temp = SkinDatas.create(session, auto_commit=True, user_id=u_id, file_name=file_name,
                                 **skindict)
    
    return temp
