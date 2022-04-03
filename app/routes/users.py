from fastapi import APIRouter
from starlette.requests import Request

from app.database.schema import Users
# from app.errors.exceptions import NotFoundUserEx
from app.models import UserMe


router = APIRouter(prefix="/user")


@router.get("/me", response_model=UserMe)
async def get_user(request: Request):
    """
    get my info
    :param request:
    :return:
    """
    user = request.state.user
    user_info = Users.get(id=user.id)
    print(user_info)
    return user_info