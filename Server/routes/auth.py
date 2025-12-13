from fastapi import APIRouter, HTTPException, Depends, Header, Request
from sqlalchemy.orm import Session
from datetime import timedelta
# import firebase_admin
# from firebase_admin import auth, credentials

from database.models import Master, HRUser
from database.conn import SessionLocal
from pydantic import BaseModel, EmailStr
from typing import List
from datetime import datetime, timedelta, timezone
from database.conn import get_db
import jwt



# cred = credentials.Certificate("path/to/serviceAccountKey.json")
# firebase_admin.initialize_app(cred)

# Base.metadata.create_all(bind=engine)
SECRET_KEY = "your-secret-key"  # Replace with your secret key
ALGORITHM = "HS256"
# 10 days
ACCESS_TOKEN_EXPIRE_MINUTES = 10 * 24 * 60

router = APIRouter()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# Schemas

class EmployeeCheckRequest(BaseModel):
    employee_id: str

class EmployeeCheckResponse(BaseModel):
    exists: bool
class LoginUser(BaseModel):
    email: EmailStr

class RegisterUser(BaseModel):
    email: EmailStr
    emp_id: str
    name: str
    password: str

class OAuthUser(BaseModel):
    email: EmailStr
    emp_id: str
    name: str
    isRegistration: bool

class HRLoginRequest(BaseModel):
    email: str
    password: str

class HRLoginResponse(BaseModel):
    access_token: str
    token_type: str

# Function to generate a JWT token with employee id as payload
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_user(token: str):
    """
    Verifies the JWT and returns the decoded claims.
    """
    db=next(get_db())
    try:
        token= token.split(" ")[1]  # Extract token from "Bearer <token>"
        decoded_claims = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        emp_id = decoded_claims.get("emp_id")
        hr_email = decoded_claims.get("hr_email")
        print(hr_email)
        role = decoded_claims.get("role")
        if role=="employee":
            user = db.query(Master).filter(Master.employee_id == emp_id).first()
            if not user:
                raise HTTPException(status_code=401, detail="Unauthorized")
            return {"emp_id": emp_id, "role": role}
        elif role=="hr":
            user = db.query(HRUser).filter(HRUser.email == hr_email).first()
            if not user:
                raise HTTPException(status_code=401, detail="Unauthorized")
            return {"hr_email": hr_email, "role": role}
        
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    finally:
        db.close()

class UserResponse(BaseModel):
    employee_id: str
    employee_name: str
    employee_email: EmailStr
    role: str
    is_selected: bool
    sentimental_score: int
    shap_values: List[str]

    class Config:
        orm_mode = True
    
@router.post("/login")
def login(user: LoginUser, db: Session = Depends(get_db)):
    db_user = db.query(Master).filter(Master.employee_email == user.email).first()
    if not db_user:
        raise HTTPException(status_code=400, detail="User does not exist")
    token= create_access_token(data={"emp_id": db_user.employee_id , "role":"employee"})
    return {"token": token, "role": "employee"}


@router.post("/oauth")
def oauth(user: OAuthUser, db: Session = Depends(get_db)):
    if not user.isRegistration:
        db_user = db.query(Master).filter(Master.employee_email == user.email).first()
        if not db_user:
            raise HTTPException(status_code=400, detail="User does not exist")
        token= create_access_token(data={"emp_id": db_user.employee_id,"role":"employee"})
        return {"token": token, "role": db_user.role}
    else:
        existing_employee = db.query(Master).filter(Master.employee_id == user.emp_id).first()
        if not existing_employee:
            raise HTTPException(
            status_code=400,
            detail="Invalid employee_id. Please contact HR."
        )
    
        # If an email is already registered for that employee, reject the registration
        if existing_employee.employee_email and existing_employee.employee_email.strip() != "":
            raise HTTPException(
                status_code=400,
                detail="This employee is already registered."
            )
        
        check_email = db.query(Master).filter(Master.employee_email == user.email).first()
        if check_email:
            raise HTTPException(
                status_code=400,
                detail="This email is already registered with another employee."
            )
        
        # Update the existing record with name and email
        existing_employee.employee_name = user.name
        existing_employee.employee_email = user.email
        db.commit()
        db.refresh(existing_employee)
        access_token= create_access_token(data={"emp_id": db_user.employee_id,"role":"employee"})
        return {"token": access_token}

@router.get("/check/{employee_id}", response_model=EmployeeCheckResponse)
def check_employee_id(employee_id: str, db: Session = Depends(get_db)):
    """
    Check if the employee_id exists in the system.
    """
    existing = db.query(Master).filter(Master.employee_id == employee_id).first()
    return {"exists": bool(existing)}

@router.post("/register")
def register(user: RegisterUser, db: Session = Depends(get_db)):
    """
    Register a new user using Firebase (or an external provider). The employee IDs are
    preloaded in the database. When a user registers, update the existing record with the
    provided name and email. If the employee_id does not exist or the email is already set,
    return an error.
    """
    # Check if the employee_id exists in the database
    existing_employee = db.query(Master).filter(Master.employee_id == user.emp_id).first()
    if not existing_employee:
        raise HTTPException(
            status_code=400,
            detail="Invalid employee_id. Please contact HR."
        )
    
    # If an email is already registered for that employee, reject the registration
    if existing_employee.employee_email and existing_employee.employee_email.strip() != "":
        raise HTTPException(
            status_code=400,
            detail="This employee is already registered."
        )
    check_email = db.query(Master).filter(Master.employee_email == user.email).first()
    if check_email:
        raise HTTPException(
                status_code=400,
                detail="This email is already registered with another employee."
            )
    
    # Update the existing record with name and email
    existing_employee.employee_name = user.name
    existing_employee.employee_email = user.email
    existing_employee.password = user.password
    db.commit()
    db.refresh(existing_employee)
    access_token = create_access_token(data={"emp_id": user.emp_id,"role":"employee"})
    return {"token": access_token}


@router.get("/employee")
def get_employee(request:Request,db: Session = Depends(get_db)):
    """
    Fetches the employee details using the token.
    """
    try:
        token = request.headers["Authorization"]
        user_data = verify_user(token)  
        emp_id = user_data["emp_id"]
        user = db.query(Master).filter(Master.employee_id == emp_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user 
    except IndexError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")


@router.post("/hr-login")
def hr_login(request: HRLoginRequest, db: Session = Depends(get_db)):
    hr_user = db.query(HRUser).filter(HRUser.email == request.email).first()
    if not hr_user or hr_user.password != request.password:
        raise HTTPException(status_code=400, detail="Invalid Creds")

    # Create the JWT token for the authenticated HR user
    token = create_access_token(data={"hr_email": hr_user.email,"role":"hr" })

    return {"token": token, "role": "hr"}


# def verify_hr(token: str, db: Session = Depends(get_db)) -> dict:
#     """
#     Verifies the JWT and returns the decoded claims.
#     """
#     try:

#         decoded_claims = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

#         hr_email = decoded_claims.get("hr_email")
#         role = decoded_claims.get("role")

#         if not hr_email or not role:
#             raise HTTPException(status_code=401, detail="Invalid token payload")

#         return {"hr_email": hr_email, "role": role}

#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.JWTError:
#         raise HTTPException(status_code=401, detail="Invalid token")

@router.get("/hr")
def get_hr(request:Request, db: Session = Depends(get_db)):
    try:
        token = request.headers["Authorization"]
        hr_data = verify_user(token)
        print(hr_data)
        hr_id = hr_data["hr_email"]
        hr = db.query(HRUser).filter(HRUser.email == hr_id).first()
        if not hr:
            raise HTTPException(status_code=404, detail="User not found")
        return hr
    except IndexError:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
        