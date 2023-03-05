# from datetime import datetime
from enum import Enum
# from mailbox import NotEmptyError
from typing import List

# from pydantic import Field
from pydantic.main import BaseModel
# from pydantic.networks import EmailStr, IPvAnyAddress


class UserRegister(BaseModel):
    email: str = None
    pw: str = None


class SnsType(str, Enum):
    email: str = "email"
    facebook: str = "facebook"
    google: str = "google"
    kakao: str = "kakao"


class Token(BaseModel):
    Authorization: str = None


class EmailRecipients(BaseModel):
    name: str
    email: str


class SendEmail(BaseModel):
    email_to: List[EmailRecipients] = None


class UserToken(BaseModel):
    id: int
    email: str = None
    name: str = None
    phone_number: str = None
    profile_img: str = None
    sns_type: str = None

    class Config:
        orm_mode = True


class UserMe(BaseModel):
    id: int
    email: str = None
    name: str = None
    phone_number: str = None
    profile_img: str = None
    sns_type: str = None

    class Config:
        orm_mode = True


class skin(BaseModel):
    wrinkle: int = None
    skin_tone: int = None
    pore_detect: int = None
    dead_skin: int = None
    oilly: int = None
    pih: int = None
    
    class Config:
        orm_mode = True

class reco(BaseModel):
    mbr_no: str = None
    recogoods1: str = None
    cossim1: float = None
    recogoods2: str = None
    cossim2: float = None
    recogoods3: str = None
    cossim3: float = None
    
    class Config:
        orm_mode = True


class MySkin(BaseModel): # 이거 안쓰면 지우기
    rank: int
    skindata: skin = None
    recommand: reco = None

    class Config:
        orm_mode = True


class skinlist(BaseModel): # 이거 안쓰면 지우기
    slist:List[MySkin] = []

    class Config:
        orm_mode = True

class cosmetic(BaseModel):
    goods_no: str = None
    goods_nm: str = None
    brand_nm: str = None
    price: int = None
    ingredient: str = None
    
    class Config:
        orm_mode = True


####---- for redis ----####

class RedisUser(BaseModel):
    email: str = None
    password: str = None

class RedisEmail(BaseModel):
    email: str = None

class RedisSkin(BaseModel):
    email: str = None
    wrinkle: str = None
    skin_tone: str = None
    pore_detect: str = None
    dead_skin: str = None
    oilly: str = None
    pih: str = None