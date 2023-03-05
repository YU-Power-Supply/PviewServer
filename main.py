from dataclasses import asdict
# from distutils.command.config import config

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from app.common.consts import EXCEPT_PATH_LIST, EXCEPT_PATH_REGEX
from app.database.conn import db, Base
from app.common.config import conf
from app.middlewares.token_vaildator import access_control
from app.middlewares.trusted_hosts import TrustedHostMiddleware
from app.routes import index, auth, users, pview, search, redis_server, auth_redis, pview_redis
# from app.routes import index

API_KEY_HEADER = APIKeyHeader(name="Authorization", auto_error=False)  # api 권한 추가


def create_app():

    c = conf()
    app = FastAPI()
    conf_dict = asdict(c)
    db.init_app(app, **conf_dict)
    # 데이터 베이스 이니셜라이즈
    Base.metadata.create_all(db.engine)
    # 미들웨어 정의
    app.add_middleware(middleware_class=BaseHTTPMiddleware, dispatch=access_control)
    app.add_middleware(  # 이부분이 없으면 백엔드와 프론트앤드 주소가 같아야 주고받을 수 있다.
        CORSMiddleware,
        allow_origins=conf().ALLOW_SITE,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=conf().TRUSTED_HOSTS, except_path=['/health'])
    # 라우터 정의
    app.include_router(index.router)
    app.include_router(redis_server.router)
    app.include_router(auth.router, tags=["Authentication"], prefix="/api")
    app.include_router(auth_redis.router, tags=["redisAuth"], prefix="/api")
    app.include_router(pview_redis.router, tags=["redisPview"], prefix="/api")
    app.include_router(search.router, tags=["Search"], prefix="/api")
    app.include_router(users.router, tags=["Users"], prefix="/api", dependencies=[Depends(API_KEY_HEADER)])
    app.include_router(pview.router, tags=["pview"], prefix="/api", dependencies=[Depends(API_KEY_HEADER)])

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=conf().PROJ_RELOAD)
