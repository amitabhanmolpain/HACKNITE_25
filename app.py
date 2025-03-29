from fastapi import FastAPI, APIRouter, HTTPException, Depends, Response
import jwt
from datetime import datetime, timedelta
from configurations import collection
from fastapi.middleware.cors import CORSMiddleware
from database.schemas import all_data
from database.models import User
from fastapi.responses import JSONResponse

app = FastAPI()
router = APIRouter()

SECRET_KEY = "your_secret_key"  # Replace this with a secure key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Set token expiration time

app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@router.get("/home")
def home():
    return "Hello world"

@router.post("/register")
async def register_user(new_user: User):

    existing_user_query = {"email": new_user.email}
    if new_user.username:
        existing_user_query["username"] = new_user.username
    existing_user = collection.find_one({"$or": [{"email": new_user.email}, {"username": new_user.username}]})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or username already exists.")
        
    try:
        new_user.hash_password() 
        resp = collection.insert_one(new_user.dict())  
        return { "message": "User registered successfully.","status_code":"200"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.post("/login")
async def login_user(user: User, response: Response):
    email = user.email
    password = user.password
    user_data = collection.find_one({"email": email})
    
    if user_data and User(**user_data).verify_password(password):
       
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
        
        
        response.set_cookie(key="access_token", value=access_token, httponly=True, max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        
        return {"message": "Login successful.", "status_code": "200"}

    raise HTTPException(status_code=401, detail="Invalid username or password.")



app.include_router(router)


