from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, func, Enum, Float
from sqlalchemy.orm import Session, relationship
from app.database.conn import Base, db


class BaseMixin:
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, nullable=False, default=func.utc_timestamp())
    
    def __init__(self):  # archemy 에서는 실행되지 않음. 다만 명시적으로 개발자들에게 알려주기 위함.
        self._q = None
        self._session = None
        self.served = None

    def all_columns(self):
        return [c for c in self.__table__.columns if c.primary_key is False and c.name != "created_at"]

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def create(cls, session: Session, auto_commit=False, **kwargs):
        """
        테이블 데이터 적재 전용 함수
        :param session:
        :param auto_commit: 자동 커밋 여부
        :param kwargs: 적재 할 데이터
        :return:
        """
        obj = cls()
        for col in obj.all_columns():
            col_name = col.name
            # print(col.name, "\n")
            if col_name in kwargs:
                setattr(obj, col_name, kwargs.get(col_name))
        session.add(obj)
        session.flush()
        if auto_commit:
            session.commit()
        return obj

    @classmethod
    def get(cls, **kwargs):  # 조회만 하는 것이므로 다른 세션을 가져와 읽고 반납한다.
        """
        Simply get a Row
        :param session:
        :param kwargs:
        :return:
        """
        session = next(db.session())
        query = session.query(cls)
        for key, val in kwargs.items():
            col = getattr(cls, key)
            query = query.filter(col == val)
        # print("\n\n", query, "\n\n")

        if query.count() > 1:
            raise Exception("Only one row is supposed to be returnd, but more than one.")
        return query.first()

    @classmethod
    def filter(cls, session: Session = None, **kwargs):
        """
        Simply get a Row
        :param session:
        :param kwargs:
        :return:
        """
        cond = []
        for key, val in kwargs.items():
            key = key.split("__")
            if len(key) > 2:
                raise Exception("No 2 more dunders")
            col = getattr(cls, key[0])

            if len(key) == 1:
                cond.append((col == val))
            elif len(key) == 2 and key[1] == 'gt':
                cond.append((col > val))
            elif len(key) == 2 and key[1] == 'gte':
                cond.append((col >= val))
            elif len(key) == 2 and key[1] == 'lt':
                cond.append((col < val))
            elif len(key) == 2 and key[1] == 'lte':
                cond.append((col <= val))
            elif len(key) == 2 and key[1] == 'in':
                cond.append((col.in_(val)))
        obj = cls()
        if session:
            obj._session = session
            obj.served = True
        else:
            obj._session = next(db.session())
            obj.served = False
        query = obj._session.query(cls)
        query = query.filter(*cond)
        obj._q = query
        return obj

    @classmethod
    def cls_attr(cls, col_name=None):
        if col_name:
            col = getattr(cls, col_name)
            return col
        else:
            return cls

    def order_by(self, *args: str):
        for a in args:
            if a.startswith("-"):
                col_name = a[1:]
                is_asc = False
            else:
                col_name = a
                is_asc = True
            col = self.cls_attr(col_name)
            self._q = self._q.order_by(col.asc()) if is_asc else self._q.order_by(col.desc())
        return self

    def update(self, auto_commit: bool = False, **kwargs):
        qs = self._q.update(kwargs)
        # get_id = self.id
        ret = None

        self._session.flush()
        if qs > 0:
            ret = self._q.first()
        if auto_commit:
            self._session.commit()
        return ret

    def delete(self, auto_commit: bool = False):
        self._q.delete()
        if auto_commit:
            self._session.commit()

    def all(self):
        result = self._q.all()
        self.close()
        return result

    def count(self):
        result = self._q.count()
        self.close()
        return result

    def close(self):
        if not self.served:
            self._session.close()
        else:
            self._session.flush()


class Users(Base, BaseMixin):
    __tablename__ = "users"
    status = Column(Enum("active", "deleted", "blocked"), default="active")
    email = Column(String(length=255), nullable=True)  # 페이스북 로그인의 경우 이메일 값 없음
    pw = Column(String(length=2000), nullable=True)
    name = Column(String(length=255), nullable=True)
    phone_number = Column(String(length=20), nullable=True)
    profile_img = Column(String(length=1000), nullable=True)
    sns_type = Column(Enum("FB", "K"), nullable=True)
    marketing_agree = Column(Boolean, nullable=True, default=True)
    updated_at = Column(DateTime, nullable=False, default=func.utc_timestamp(), onupdate=func.utc_timestamp())


class SkinDatas(Base, BaseMixin):
    __tablename__ = "skindatas"
    wrinkle = Column(Float, nullable=False)
    skin_tone = Column(Integer, nullable=False)
    pore_detect = Column(Float, nullable=False)
    dead_skin = Column(Float, nullable=False)
    oilly = Column(Enum("oilly", "normal", "dry"), nullable=False)
    pih = Column(Integer, nullable=False)

    acne = Column(Integer, nullable=False)
    whitening = Column(Integer, nullable=False)
    stimulus = Column(Integer, nullable=False)
    wrinkle = Column(Integer, nullable=False)
    moisture = Column(Integer, nullable=False)
    moisturizing = Column(Integer, nullable=False)
    oil = Column(Integer, nullable=False)

    file_name = Column(String(length=100), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship('Users', backref="skindatas")


"""
class RecoDatas(Base, BaseMixin):
    __tablename__ = "recodatas"
        s_acne = Column(Integer, nullable=True)
        s_whitening = Column(Integer, nullable=True)
        s_stimulus = Column(Integer, nullable=True)
        s_wrinkle = Column(Integer, nullable=True)
        s_moisture = Column(Integer, nullable=True)
        a_moisturizing = Column(Integer, nullable=True)
        a_oils = Column(Integer, nullable=True)
"""
