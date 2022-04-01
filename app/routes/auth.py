from datetime import datetime, timedelta

import bcrypt
import jwt
from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session
from starlette.responses import JSONResponse

from app import models
from app.common.consts import JWT_SECRET, JWT_ALGORITHM
from app.database.conn import db
from app.database.schema import Users
from app.models import SnsType, Token, UserToken


router = APIRouter(prefix="/auth")


@router.post("/register/{sns_type}", status_code=201)
async def register(sns_type: SnsType, reg_info: models.UserRegister, session: Session = Depends(db.session)):
    """
    `회원가입 API`\n
    :params sns_type:
    :param reg_info:
    :param session:
    :return:
    """

    if sns_type == SnsType.email:
        is_exist = await is_email_exist(reg_info.email)
        if not reg_info.email or not reg_info.pw:
            return JSONResponse(status_code=400, content=dict(msg="Email and PW must be provided"))
        if is_exist:
            return JSONResponse(status_code=400, content=dict(msg="EMAIL_EXISTS"))
        # gensalt() : 음식에 소금을 몇번 뿌리느냐에 따라 맛이 달라지듯 약간의 값을 추가해 해시값을 유추하기 어렵게 만든다.
        hash_pw = bcrypt.hashpw(reg_info.pw.encode("utf-8"), bcrypt.gensalt())
        Users.create(session, auto_commit=True, pw=hash_pw, email=reg_info.email)
        return JSONResponse(status_code=201, content=dict(msg="REGISTER SUCCESS"))
    return JSONResponse(status_code=400, content=dict(msg="NOT_SUPPORTED"))


@router.post("/login/{sns_type}", status_code=200, response_model=Token)
async def login(sns_type: SnsType, user_info: models.UserRegister):
    if sns_type == SnsType.email:
        is_exist = await is_email_exist(user_info.email)
        if not user_info.email or not user_info.pw:
            return JSONResponse(status_code=400, content=dict(msg="Email and PW must be provided"))
        if not is_exist:
            return JSONResponse(status_code=400, content=dict(msg="NO_MATCH_USER"))
        user = Users.get(email=user_info.email)
        is_verified = bcrypt.checkpw(user_info.pw.encode('utf-8'), user.pw.encode('utf-8'))
        if not is_verified:
            return JSONResponse(status_code=400, content=dict(msg="NO_MATCHED_USER"))
        token = dict(
            Authorization=f"Bearer \
                {create_access_token(data=UserToken.from_orm(user).dict(exclude={'pw', 'marketing_agree'}), )}")
        return token
    return JSONResponse(status_code=400, content=dict(msg="NOT_SUPPPORTED"))


async def is_email_exist(email: str):

    get_email = Users.get(email=email)
    if get_email:
        return True
    return False


def create_access_token(*, data: dict = None, expires_delta: int = None):
    to_encode = data.copy()
    if expires_delta:
        to_encode.update({"exp": datetime.utcnow() + timedelta(hours=expires_delta)})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt
