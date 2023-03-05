import redis
from fastapi import APIRouter

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)


router = APIRouter(prefix="/redis")

# Define User model
class User:
    def __init__(self, user_id: str, user_name: str, password: str):
        self.user_id = user_id
        self.user_name = user_name
        self.password = password

    def as_dict(self):
        return {
            'user_id': self.user_id,
            'user_name': self.user_name,
            'password': self.password
        }

    @classmethod
    def from_dict(cls, user_dict):
        return cls(
            user_id=user_dict[b'user_id'].decode('utf-8'),
            user_name=user_dict[b'user_name'].decode('utf-8'),
            password=user_dict[b'password'].decode('utf-8')
        )

# Define functions for user registration and login
def register_user(user_name: str, password: str):
    # Generate unique user ID
    user_id = r.incr('user_id')

    # Create user object
    user = User(user_id=str(user_id), user_name=user_name, password=password)

    # Add user to Redis
    r.hmset(f'user:{user.user_id}', user.as_dict())
    r.set(f'user_name:{user.user_name}', user.user_id)

    return user

def login_user(user_name: str, password: str):
    # Get user ID by user name
    user_id = r.get(f'user_name:{user_name}').decode('utf-8')
    if user_id is None:
        return None

    # Get user from Redis
    user_dict = r.hgetall(f'user:{user_id}')
    if not user_dict:
        return None

    user = User.from_dict(user_dict)
    # Check password
    if user.password != password:
        return None

    return user

# Define API endpoints
@router.post('/register')
async def register(user_name: str, password: str):
    user = register_user(user_name, password)
    return user.as_dict()

@router.post('/login')
async def login(user_name: str, password: str):
    user = login_user(user_name, password)
    if user is None:
        return {'message': 'Invalid username or password'}
    else:
        return user.as_dict()
