from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import json
from dotenv import load_dotenv
import os
from question_bank import question_bank
from gemini import generate_text
from database.conn import get_db
from typing import List, Dict
from datetime import datetime
from chatgpt import chat_with_gpt4o
import httpx
import io
from fastapi.responses import StreamingResponse
from .auth import verify_user
from .message import chatbot_conversation,retrieve_relevant_questions,generate_user_summary


from database.models import Conversation,Message, Master


load_dotenv()
API_KEY = os.getenv("GENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

router = APIRouter()

class StartConversationRequest(BaseModel):
    employee_name: str
    employee_id: str
    shap: List[str]

class MessageRequest(BaseModel):
    conversation_id:int
    message:str
    message_type:str
    chat_history: List[Dict[str, str]]
    question_set: List[str] 


class PromptRequest(BaseModel):
    prompt: str


@router.get("/start")
async def start_conversation(request: Request, db: Session = Depends(get_db)):
    try:
        token=request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_data=verify_user(token)
        emp_id,role=user_data["emp_id"],user_data["role"]
        user=db.query(Master).filter(Master.employee_id == emp_id).first()
        if role != "employee":
            raise HTTPException(status_code=401, detail="Unauthorized access")
        system_prompt="You are a friendly and professional HR assistant designed to check in on employees in a warm and concise manner. Always keep the tone polite, supportive, and under 2 lines."
        user_prompt=f"The employee's name is {user.employee_name}. Greet her and let her know this is a regular check-in to see how she’s doing today. Keep it short and caring."
        greeting_message = chat_with_gpt4o(system_prompt,user_prompt)
        new_message = Message(
            content=greeting_message,
            sender_type="chatbot",
            message_type="welcome"
        )
        db.add(new_message)
        db.commit()
        db.refresh(new_message)

        # Pre-select the questions based on `shap` topics
       

        # if not selected_questions:
        #     raise HTTPException(status_code=400, detail="No valid questions found for the given SHAP topics")

        # now = datetime.now()
        new_conversation = Conversation(
            employee_id=user.employee_id,
            employee_name=user.employee_name,
            message_ids=[new_message.id],  # Store the message ID
            # date=now.date(),
            # time=now.time()
        )
        db.add(new_conversation)
        db.commit()
        db.refresh(new_conversation)
        return {"chatbot_response":greeting_message , "conversation_id":new_conversation.id}   # Send the conversation id along with the message
    except:
        db.rollback()
        raise HTTPException(status_code=500, detail="Error starting conversation")


@router.post("/message")
async def send_message(request:Request,db: Session = Depends(get_db)):
    """
    Accepts employee message, generates chatbot response, and appends both
    message IDs to the existing conversation using `conversation_id`.
    """
    try:
        # Retrieve existing conversation using `conversation_id`
        print("Inside /message endpoint")
        token=request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_data=verify_user(token)
        emp_id,role=user_data["emp_id"],"employee"
        user=db.query(Master).filter(Master.employee_id == emp_id).first()
        if role != "employee":
            print("Unauthorized access attempt")
            raise HTTPException(status_code=401, detail="Unauthorized access")
        # how to get the body from the request in the form of MessageRequest
        body = await request.json()

        data = MessageRequest(**body)
        conversation = db.query(Conversation).filter_by(id=data.conversation_id).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get the relevant questions based on the user's SHAP values
        relevant_questions = []
        if data.question_set==[]:
            question_set = []
            selected_questions = {}
            for topic in user.shap_values:
                if topic in question_bank:
                    selected_questions.update({topic: question_bank[topic]})
            for key, value in selected_questions.items():
                # question_set.append(f"{key}: {value['description']}")
                for i, question in enumerate(value['questions'], 1):
                    question_set.append(f"{question}")
            # user_summary=generate_user_summary(user.shap_nature)

            # Convert shap_values dict to a readable string summary
            # if isinstance(user.shap_values, dict):
            #     user_summary = ", ".join([f"{k}: {v}" for k, v in user.shap_values.items()])
            # else:
            #     user_summary = ""

            user_summary = ""
            relevant_questions=retrieve_relevant_questions(user_summary,question_set)
        else:
            relevant_questions=data.question_set

        user_summary=""
        
        
        if data.message_type=="welcome":
            # question_set
            print("Shap",shap_values:=user.shap_values)
            generated_message,message_type=chatbot_conversation(user.shap_values,[],"",data.message_type,relevant_questions)
            # Store the AI response in `Message` table
            print("Generated Message:",generated_message)
            chatbot_message = Message(
                content=generated_message,
                sender_type="chatbot",
                message_type=message_type
            )
            db.add(chatbot_message)
            db.commit()
            db.refresh(chatbot_message)
            return {
                "chatbot_response": generated_message,
                "message_type":message_type,
                "question_set": question_set
            }

        # Store the employee's message in `Message` table
        employee_message = Message(
            content=data.message,
            sender_type="employee",
            message_type="user_msg"
        )
        db.add(employee_message)
        db.commit()
        db.refresh(employee_message)


        print("Stored Employee Message ID:",employee_message.id)

        # Retrieve the chat-history
        chat_history  = data.chat_history
        chat_history_text = "\n".join([
            f"{msg['sender_type'].capitalize()}: {msg['message']}"
            for msg in chat_history
        ])

        generated_message,message_type=chatbot_conversation(user.shap_values,chat_history_text,data.message,data.message_type,relevant_questions)
        # generated_message = generated_message(selected_questions,chat_history_text, data.message)

        # Store the AI response in `Message` table
        print("Generated Message:",generated_message)
        chatbot_message = Message(
            content=generated_message,
            sender_type="chatbot",
            message_type=message_type
        )
        db.add(chatbot_message)
        db.commit()
        db.refresh(chatbot_message)

        # Append both message IDs to the existing conversation
        conversation.message_ids.append(employee_message.id)
        conversation.message_ids.append(chatbot_message.id)
        db.commit()
    
        return {
            "chatbot_response": generated_message,
            "message_type":message_type,
            "question_set": relevant_questions
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/history/employee")
def get_conversation_history(request:Request,db:Session = Depends(get_db)):
    try:
        token=request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_data=verify_user(token)
        emp_id,role=user_data["emp_id"],"employee"
        # user=db.query(Master).filter(Master.employee_id == emp_id).first()
        if role != "employee":
            raise HTTPException(status_code=401, detail="Unauthorized access")
        
        conversations = db.query(Conversation).filter(Conversation.employee_id == emp_id).all()
        if not conversations:
            return []
        return [
            {
                "conversation_id": conv.id,
                # "employee_id": conv.employee_id,
                # "employee_name": conv.employee_name,
                "date": conv.date,
                "time": conv.time,
            } for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetcging the conversations: {str(e)}")


@router.get("/history/{conversation_id}")
#how to get the token too
def get_messages(conversation_id:str,request:Request,db:Session=Depends(get_db)):  # Ensure ID is int if it's an integer column
    try:
        # Fetch the conversation by ID
        token=request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_data=verify_user(token)
        emp_id,role=user_data["emp_id"],"employee"
        if role != "employee":
            raise HTTPException(status_code=401, detail="Unauthorized access")
        
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if conversation.employee_id != emp_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to this conversation")

        # Fetch associated messages by their IDs
        messages = db.query(Message).filter(Message.id.in_(conversation.message_ids)).all()

        # Format the response
        message_list = [{"id": msg.id, "content": msg.content, "sender_type": msg.sender_type,"time":msg.time,"message_type":msg.message_type} for msg in messages]

        return message_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Returns users whose reports are generated today
@router.get("/todays_reports")
def fetch_todays_conv(db: Session = Depends(get_db)):
    try:
        today = datetime.today().date()
        # Get conversations with today's date and report not empty
        conversations = db.query(Conversation).filter(
            Conversation.date == today,
            Conversation.report != ""
        ).all()

        combined_data = []
        for conv in conversations:
            employee = db.query(Master).filter(Master.employee_id == conv.employee_id).first()
            if employee:
                combined_data.append({
                    "Employee_ID": employee.employee_id,
                    "Employee_Name": employee.employee_name,
                    "Employee_Email": employee.employee_email,
                    "Employee_Role":employee.role,
                    "Is_Selected":employee.is_selected,
                    "Is_Flagged":employee.is_Flagged,
                    "Report": conv.report,
                    "Feature_Vector":employee.feature_vector,
                    "Conversation_Completed":employee.conversation_completed,
                    "Sentimental_Score":employee.sentimental_score,
                    })
        return combined_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights/{conversation_id}")
def get_insights(conversation_id: int, request:Request,db: Session = Depends(get_db)):
    """
    Generate insights based on the entire conversation using Gemini.
    """
    try:
        # 1. Fetch the conversation by ID
        token=request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user_data=verify_user(token)
        emp_id,role=user_data["emp_id"],user_data["role"]
        if role != "employee":
            raise HTTPException(status_code=401, detail="Unauthorized access")
        user=db.query(Master).filter(Master.employee_id == emp_id).first()
        conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        if conversation.employee_id != emp_id:
            raise HTTPException(status_code=403, detail="Unauthorized access to this conversation")

        # 2. Fetch all messages associated with this conversation
        messages = db.query(Message).filter(Message.id.in_(conversation.message_ids)).all()

        if not messages:
            raise HTTPException(status_code=404, detail="No messages found for this conversation")

        # 3. Compile the conversation history
        conversation_history = ""
        for msg in messages:
            role = "Employee" if msg.sender_type == "employee" else "Chatbot"
            conversation_history += f"{role}: {msg.content}\n"

        system_prompt = ("You are an empathetic and analytical assistant. Your task is to carefully analyze workplace conversations and generate thoughtful, structured insights. Focus on identifying mood, concerns, key issues, and offer actionable suggestions for improvement. Present your response clearly and professionally.")
        # 4. Create an insightful prompt for openai
        insight_prompt = (
            f"Here is a conversation between the employee ({conversation.employee_name}) and a chatbot:\n\n"
            f"{conversation_history}\n\n"
            f"Generate detailed insights based on this conversation:\n"
            f"- Identify the employee's mood, concerns, and sentiments.\n"
            f"- Highlight key issues or recurring themes.\n"
            f"- Provide suggestions or recommendations to improve the situation.\n"
            f"- Format the insights in a clear and organized manner."
        )
        # 5. Generate insights using Gemini
        insights = chat_with_gpt4o(system_prompt, insight_prompt)

        chatbot_message = Message(
            content=insights,
            sender_type="chatbot",
            message_type="insight"
        )
        db.add(chatbot_message)
        db.commit()
        db.refresh(chatbot_message)

        conversation.message_ids.append(chatbot_message.id)
        db.commit()
        db.refresh(conversation)

        user.conversation_completed=True
        db.commit()
        db.refresh(user)

        # 6. Return the insights
        return {
            "insights": insights
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

@router.post("/test")
def generate(request: PromptRequest):
    try:
        response = generate_text(request.prompt)
        return {"response": response}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")   
    

@router.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Convert speech audio to text using Deepgram"""
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="Deepgram API key not configured") 
    try:
        audio_content = await audio.read()
        # Send to Deepgram
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
                headers={
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/wav"  # Adjust based on your audio format
                },
                content=audio_content
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail="Failed to transcribe audio")
            
            result = response.json()

            transcript = result["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            return {"transcript": transcript}
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in speech-to-text: {str(e)}")
    
@router.post("/tts")
async def text_to_speech(request: PromptRequest):
    """
    Convert text to speech using Deepgram API
    
    Takes text input and returns audio stream
    """
    if not DEEPGRAM_API_KEY:
        raise HTTPException(status_code=500, detail="Deepgram API key not configured")
    try:
        payload = {
            "text": request.prompt,
        }
        # Set up headers for Deepgram API request
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        # Make the request to Deepgram
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.deepgram.com/v1/speak?model=aura-asteria-en&gender=female",
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            # Check for successful response
            if response.status_code != 200:
                error_message = f"Deepgram TTS API error: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_message += f" - {error_detail}"
                except:
                    pass
                
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_message
                )
            
            # # Return the audio stream
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="audio/mp3",
                headers={
                    "Content-Disposition": f"attachment; filename=speech.mp3"
                }
            )
    
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Deepgram: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing text-to-speech: {str(e)}")
