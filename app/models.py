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
    wrinkle: float = None
    skin_tone: int = None
    pore_detect: float = None
    dead_skin: float = None
    oilly: str = None
    pih: int = None


class reco(BaseModel):
    acne: int = None
    whitening: int = None
    stimulus: int = None
    wrinkle: int = None
    moisture: int = None
    moisturizing: int = None
    oil: int = None


class MySkin(BaseModel):
    id: int
    skindata: skin = None
    recomand: reco = None

    class Config:
        orm_mode = True
