from fastapi import APIRouter, Depends, File, UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

import cv2
import numpy as np
import faiss
import pandas as pd

from app.database.schema import SkinDatas, Recommandation, Cosmetics
from app import models
from app.database.conn import rdb
from app.pview_core import Oilly, PIH, Pore, SkinTone, Wrinkle, recommand, DeadSkin
from app.pview_core.skin_cropper import skin_cropper


from datetime import datetime
import secrets


router = APIRouter(prefix="/rpview")

user_vectors_df = pd.read_excel('app/pview_core/userreco.xlsx', index_col=0)
user_vectors_search = user_vectors_df.reset_index()
user_vectors_df = user_vectors_df.set_index('mbrNo')
user_vectors = user_vectors_df.to_numpy(dtype=np.float32)


user_reco_df = pd.read_excel('app/pview_core/recolist.xlsx', index_col=0)
user_reco_df = user_reco_df.set_index('mbr_no', drop=True)

# Set up Faiss index
d = 6  # dimensionality of the vectors
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index on the user vectors
index.train(user_vectors)

# Add the user vectors to the index
index.add(user_vectors)


def from_dict(skin_dict):

    return {
        "email" : skin_dict[b'email'].decode('utf-8'),
        "wrinkle" : skin_dict[b'wrinkle'].decode('utf-8'),
        "skin_tone" : skin_dict[b'skin_tone'].decode('utf-8'),
        "pore_detect" : skin_dict[b'pore_detect'].decode('utf-8'),
        "dead_skin" : skin_dict[b'dead_skin'].decode('utf-8'),
        "oilly" : skin_dict[b'oilly'].decode('utf-8'),
        "pih" : skin_dict[b'pih'].decode('utf-8')
    }

def to_dict(skin_dict):

    return {
        "email" : skin_dict.email,
        "wrinkle" : skin_dict.wrinkle,
        "skin_tone" : skin_dict.skin_tone,
        "pore_detect" : skin_dict.pore_detect,
        "dead_skin" : skin_dict.dead_skin,
        "oilly" : skin_dict.oilly,
        "pih" : skin_dict.pih,
    }



@router.post("/globalskin", status_code=201, response_model=models.RedisSkin)
async def post_skin(file: UploadFile = File(...)):
    """
    {run Skin}
    :param request:
    """
    user, extension = file.filename.split(".")
    if not extension in ("jpg", "jpeg", "png"):
        return JSONResponse(status_code=400, content=dict(msg="image must be jpg or png format"))

    byte_file = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(byte_file, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (540, 720))
    
    try :
        _, masked_img = skin_cropper(img, 'all')
    except:
        return JSONResponse(status_code=400, content=dict(msg="it is not a face image."))

    masked_img_256 = cv2.resize(masked_img, (256, 256))

    skindict = {"skin_tone" : str(SkinTone.skinToneDetect(masked_img_256)),
                "pore_detect" : str(Pore.poreDetect(masked_img_256)),
                "pih" : str(PIH.detect_pih(masked_img_256, "app/pview_core/weights/pih_model/pih_model_weight_0907.h5" ))}
    
    rdb.hmset(user, skindict)
    return skindict


@router.post("/detailskin", status_code=201, response_model=models.RedisSkin)
async def post_skin(file: UploadFile = File(...)):
    """
    {run Skin}
    :param request:
    """
    user, extension = file.filename.split(".")
    if not extension in ("jpg", "jpeg", "png"):
        return JSONResponse(status_code=400, content=dict(msg="image must be jpg or png format"))

    byte_file = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(byte_file, cv2.IMREAD_COLOR)


    skindict = {"wrinkle" : str(Wrinkle.wrinkleDetect(img)),
                "dead_skin" : str(DeadSkin.detect_deadskin(img, ""))}
    
    rdb.hmset(user, skindict)
    return skindict



@router.post("/recommand") # 화장품 추천받기
async def cosmetic_recommand(email: models.RedisEmail):
    s_data = from_dict(rdb.hgetall(email.email))

    if not s_data:
        return JSONResponse(status_code=400, content=dict(msg="NO_SKINDATA!!"))
    
    s_vector = np.array([s_data["oilly"], s_data["dead_skin"], s_data["pih"], s_data["wrinkle"], 
                         s_data["pore_detect"], s_data["skin_tone"]], dtype=np.float32)
    
    distances, indices = index.search(np.array([s_vector]), 3)
    
    recolist = []
    recodict = {"length" : 3, "cosmeticlist" : []}
    
    for i, idx in enumerate(indices[0]):
        mbr = user_vectors_search.iloc[idx, 1]
        dist = 1.0 / (1.0 + distances[0][i])
        temp = user_reco_df.loc[mbr]
        recolist += [(temp.recogoods1, float(temp.cossim1)*float(dist)), (temp.recogoods2, float(temp.cossim2)*float(dist)), (temp.recogoods3, float(temp.cossim3)*float(dist))]
    
    recolist.sort(key=lambda x:x[1], reverse=True)
    temp = Cosmetics.filter(goods_no__in=[list(l) for l in zip(*recolist[:3])][0]).all(limit=3)
    for reco in temp:
        recodict["cosmeticlist"].append({"goods_nm" : reco.goods_nm, "brand_nm" : reco.brand_nm, "category" : reco.category, "price" : reco.price, "ingredient" : reco.ingredient})

    return recodict



@router.post("/pushskindata", status_code=201, response_model=models.RedisSkin) # 강제로 피부데이터 삽입
async def push_skindata(skin_list: models.RedisSkin):
    skindict =  to_dict(skin_list)
    rdb.hmset(skin_list.email, skindict)
    return skindict

    
    
    
def input_skindata(skindict: dict):

    return True


