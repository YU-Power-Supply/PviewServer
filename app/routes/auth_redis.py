from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app import models
from app.common.consts import JWT_SECRET, JWT_ALGORITHM
from app.database.conn import rdb

from app.models import SnsType, Token


#for only didimdol

router = APIRouter(prefix="/rauth")

# Define User model


def from_dict(user_dict):
    
    return {
        "email" : user_dict[b'email'].decode('utf-8'),
        "password" : user_dict[b'password'].decode('utf-8')   
    }




@router.post("/register/{sns_type}", status_code=201)
async def register(sns_type: SnsType, reg_info: models.RedisUser):
    """
    `회원가입 API`\n
    :params sns_type:
    :param reg_info:
    :param session:
    :return:
    """

    if sns_type == SnsType.email:
        is_exist = await is_email_exist(reg_info.email)
        if not reg_info.email or not reg_info.password:
            return JSONResponse(status_code=400, content=dict(msg="Email and PW must be provided"))
        if is_exist:
            return JSONResponse(status_code=202, content=dict(msg="EMAIL_EXISTS"))
        # gensalt() : 음식에 소금을 몇번 뿌리느냐에 따라 맛이 달라지듯 약간의 값을 추가해 해시값을 유추하기 어렵게 만든다.
        password = bcrypt.hashpw(reg_info.password.encode("utf-8"), bcrypt.gensalt())
        rdb.hmset(reg_info.email, {"email" : reg_info.email, "password" : password})
        return JSONResponse(status_code=201, content=dict(msg="REGISTER SUCCESS"))
    return JSONResponse(status_code=400, content=dict(msg="NOT_SUPPORTED"))



@router.post("/login/{sns_type}", status_code=200, response_model=Token)
async def login(sns_type: SnsType, user_info: models.RedisUser):
    if sns_type == SnsType.email:
        is_exist = await is_email_exist(user_info.email)
        if not user_info.email or not user_info.password:
            return JSONResponse(status_code=400, content=dict(msg="Email and PW must be provided"))
        if not is_exist:
            return JSONResponse(status_code=400, content=dict(msg="NO_MATCH_USER"))
        db_user = from_dict(rdb.hgetall(user_info.email))
        is_verified = bcrypt.checkpw(user_info.password.encode('utf-8'), db_user["password"].encode('utf-8'))
        if not is_verified:
            return JSONResponse(status_code=400, content=dict(msg="NO_MATCHED_USER"))
        return JSONResponse(status_code=201, content=dict(msg="LOGIN SUCCESS"))
    return JSONResponse(status_code=400, content=dict(msg="NOT_SUPPPORTED"))


async def is_email_exist(email: str):

    get_email = rdb.hgetall(email)

    if get_email:
        return True
    return False
