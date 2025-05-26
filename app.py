# 1. IMPORTS
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # Sửa import AdamW từ torch.optim
import os, uvicorn, logging, re, random, asyncio, json
from datetime import datetime
import redis.asyncio as redis
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import numpy as np
from functools import lru_cache
from config.config import Config
from utils.data_collector import DataCollector
from training.fine_tune import train_model, ChatbotDataset

# Thêm vào phần imports
templates = Jinja2Templates(directory="templates")

# 3. LOGGING SETUP
def setup_logging():
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / f'chatbot_{datetime.now().strftime("%Y%m%d")}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

setup_logging()

# 4. PYDANTIC MODELS
class ChatInput(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=500)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    language: str = Field(default="vi")
    context: Optional[Dict[str, str]] = None

    @validator('user_input')
    def validate_input(cls, v):
        if not v.strip():
            raise ValueError('Input không được để trống')
        return v.strip()

# 5. APP INITIALIZATION
app = FastAPI(
    title="Chatbot API",
    description="API cho Chatbot AI sử dụng transformer",
    version="1.0.0"
)

# 6. MIDDLEWARE SETUP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thêm sau phần MIDDLEWARE SETUP
app.mount("/static", StaticFiles(directory="static"), name="static")

# 7. REDIS SETUP
redis_client = None
USE_REDIS = False  # Thêm flag để kiểm soát việc sử dụng Redis
FALLBACK_RESPONSES = [
    "Xin lỗi, tôi đang gặp sự cố kết nối.",
    "Vui lòng thử lại sau một lát.",
    "Hệ thống đang bận, xin đợi giây lát.",
    "Tôi không hiểu rõ câu hỏi của bạn, bạn có thể diễn đạt lại được không?"
]

@app.on_event("startup")
async def startup_event():
    # Tạo thư mục cần thiết
    Config.create_directories()
    
    # Khởi tạo Redis nếu cần
    if USE_REDIS:
        try:
            global redis_client
            redis_client = redis.from_url(
                "redis://localhost",
                encoding="utf-8", 
                decode_responses=True
            )
            await FastAPILimiter.init(redis_client)
            logging.info("✅ Kết nối Redis thành công")
        except Exception as e:
            logging.error(f"❌ Lỗi kết nối Redis: {str(e)}")
            redis_client = None

    # Training định kỳ
    async def scheduled_training():
        try:
            quality_convs = await data_collector.get_quality_conversations(
                min_score=Config.COLLECTOR_CONFIG['quality_thresholds']['min_quality_score']
            )
            if quality_convs:
                await train_model(quality_convs, model_manager.model_dir)
            await asyncio.sleep(24 * 60 * 60)
        except Exception as e:
            logging.error(f"Scheduled training error: {str(e)}")
            await asyncio.sleep(60)  # Đợi 1 phút nếu có lỗi
    
    asyncio.create_task(scheduled_training())

# 8. MODEL SETUP
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Sửa đường dẫn để trỏ tới thư mục results
        self.model_dir = Path("training/results")  # Thay đổi này
        self.model = None
        self.tokenizer = None
        logging.info(f"Using device: {self.device}")
    
    async def load_model(self):
        """Load model từ thư mục results"""
        try:
            logging.info("Loading tokenizer from results...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                local_files_only=True,
                trust_remote_code=True
            )
            
            logging.info("Loading model from results...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_dir,
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)
            
            self.model.eval()  # Đặt model ở chế độ evaluation
            logging.info(f"✅ Model loaded successfully from results on {self.device}")
            return True
            
        except Exception as e:
            logging.error(f"❌ Model loading error: {str(e)}")
            return False

    async def fine_tune(self, train_data: List[Dict]):
        try:
            await train_model(train_data, self.model_dir)
            logging.info(f"Fine-tuned on {len(train_data)} samples")
        except Exception as e:
            logging.error(f"Fine-tuning error: {str(e)}")

model_manager = ModelManager()

# Thêm sau phần khởi tạo ModelManager
data_collector = DataCollector()

# 9. UTILITY FUNCTIONS
def preprocess_question(text: str) -> str:
    """Tiền xử lý câu hỏi"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s?!,.]', '', text)
    text = re.sub(r'\s+', ' ', text)
    if any(q in text for q in ['ai', 'làm sao', 'như thế nào', 'gì', 'bao giờ', 'ở đâu']):
        if not text.endswith('?'):
            text += '?'
    return text

def get_intent(text: str) -> str:
    """Phân tích ý định chi tiết"""
    text = text.lower()
    intents = {
        'greeting': ['xin chào', 'hello', 'hi', 'chào', 'alo'],
        'farewell': ['tạm biệt', 'bye', 'goodbye', 'gặp lại'],
        'question': ['là gì', 'như thế nào', 'bao giờ', 'thế nào', 'làm sao', 'tại sao'],
        'thanks': ['cảm ơn', 'thanks', 'thank', 'cảm tạ'],
        'agreement': ['đồng ý', 'được', 'ok', 'ừ', 'vâng'],
        'disagreement': ['không', 'chưa', 'không đồng ý'],
        'help': ['giúp', 'hỗ trợ', 'support'],
        'complain': ['khiếu nại', 'phàn nàn', 'không hài lòng']
    }
    
    for intent, patterns in intents.items():
        if any(p in text for p in patterns):
            return intent
    return 'other'

def enhance_context(history: List[Dict], current_input: str) -> Dict[str, Any]:
    """Tạo ngữ cảnh phong phú"""
    current_time = datetime.now()
    time_context = (
        "buổi sáng" if 5 <= current_time.hour < 12
        else "buổi chiều" if 12 <= current_time.hour < 18
        else "buổi tối"
    )
    
    conversation_state = "follow_up" if history else "new"
    
    topics = {
        'tech': ['máy tính', 'điện thoại', 'phần mềm', 'ứng dụng'],
        'general': ['thời tiết', 'tin tức', 'thể thao'],
        'personal': ['bạn', 'tôi', 'chúng ta']
    }
    
    current_topic = next(
        (topic for topic, keywords in topics.items() 
         if any(keyword in current_input.lower() for keyword in keywords)),
        'unknown'
    )
    
    context = {
        "time": time_context,
        "datetime": current_time.isoformat(),
        "intent": get_intent(current_input),
        "conversation_state": conversation_state,
        "topic": current_topic,
        "preprocessed_input": preprocess_question(current_input)
    }
    
    if history:
        last_exchange = history[-1]
        context.update({
            "last_question": last_exchange['user'],
            "last_response": last_exchange['bot'],
            "conversation_length": len(history)
        })
    
    return context

async def generate_response(
    text: str,
    context: Dict[str, Any],
    intent: str,
    temperature: float = 0.7
) -> str:
    """Generate câu trả lời tự nhiên"""
    try:
        if not model_manager.model:
            if not await model_manager.load_model():
                return random.choice(FALLBACK_RESPONSES)
        
        # Tạo prompt đơn giản hơn
        prompt = f"chat: {text}"
        
        # Tokenize với padding và truncation
        inputs = model_manager.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128  # Giới hạn độ dài input
        ).to(model_manager.device)

        # Generate với các tham số phù hợp
        with torch.no_grad():
            outputs = model_manager.model.generate(
                **inputs,
                max_length=128,        # Giới hạn độ dài output
                min_length=10,         # Độ dài tối thiểu
                num_beams=3,           # Giảm beam search
                repetition_penalty=1.2, # Giảm lặp lại
                length_penalty=1.0,
                early_stopping=True,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.9
            )

        # Decode và làm sạch response
        response = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.strip()
        
        # Thêm dấu câu nếu cần
        if not response.endswith(('.', '!', '?')):
            response += '.'
            
        # Log để debug
        logging.info(f"Input: {text}")
        logging.info(f"Response: {response}")
            
        return response
        
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        return random.choice(FALLBACK_RESPONSES)

# 10. API ENDPOINTS
@app.post("/chat")
async def chat_response(
    chat_input: ChatInput,
    background_tasks: BackgroundTasks
):
    """Main chat endpoint"""
    try:
        user_text = preprocess_question(chat_input.user_input)
        intent = get_intent(user_text)
        context = enhance_context(chat_input.conversation_history, user_text)
        
        # Handle basic intents
        if intent == 'greeting':
            time_context = context['time']
            responses = [
                f"Xin chào! Chúc bạn có một {time_context} tốt lành!",
                f"Chào bạn! Rất vui được gặp bạn trong {time_context} này.",
                f"Xin chào! Tôi có thể giúp gì cho bạn trong {time_context} này?"
            ]
            response = random.choice(responses)
        elif intent == 'farewell':
            response = random.choice([
                "Tạm biệt! Hẹn gặp lại bạn sau nhé!",
                "Chào tạm biệt! Rất vui được trò chuyện với bạn!",
                "Goodbye! Chúc bạn có một ngày tốt lành!"
            ])
        elif intent == 'thanks':
            response = random.choice([
                "Không có gì! Rất vui được giúp bạn.",
                "Không có chi! Đó là nhiệm vụ của tôi.",
                "Rất vui vì đã giúp được bạn!"
            ])
        else:
            response = await generate_response(user_text, context, intent)
        
        # Lưu cuộc hội thoại
        quality_score = data_collector.evaluate_conversation_quality(
            user_text, 
            response,
            context
        )
        
        background_tasks.add_task(
            data_collector.save_conversation,
            user_text=user_text,
            bot_response=response,
            intent=intent,
            context=context,
            quality_score=quality_score
        )
        
        # Update history
        updated_history = chat_input.conversation_history + [
            {"user": user_text, "bot": response}
        ][-Config.MAX_HISTORY:]
        
        result = {
            "response": response,
            "conversation_history": updated_history,
            "intent": intent,
            "context": context
        }
        
        # Chỉ cache nếu có Redis
        if USE_REDIS and redis_client:
            try:
                cache_key = f"chat:{hash(user_text)}"
                background_tasks.add_task(
                    redis_client.set,
                    cache_key,
                    json.dumps(result),
                    ex=Config.CACHE_TTL
                )
            except Exception as e:
                logging.error(f"Caching error: {str(e)}")
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return JSONResponse(
            content={
                "response": random.choice(FALLBACK_RESPONSES),
                "conversation_history": chat_input.conversation_history,
                "intent": "error"
            },
            status_code=500
        )

# Thêm route mới cho trang chủ
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve trang chủ"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# Thêm route xử lý lỗi
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Không tìm thấy endpoint này",
            "detail": "Vui lòng kiểm tra lại URL"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        memory_info = {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved()
        } if torch.cuda.is_available() else None
        
        return {
            "status": "healthy",
            "model": {
                "name": str(model_manager.model_dir),
                "device": str(model_manager.device),
                "loaded": model_manager.model is not None
            },
            "memory": memory_info,
            "redis": await redis_client.ping() if redis_client else False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

# 11. SHUTDOWN EVENT
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup khi shutdown"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if redis_client:
        await redis_client.close()

# 12. MAIN
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

async def train_from_quality_conversations():
    """Training model từ các hội thoại chất lượng cao"""
    # Lấy conversations chất lượng tốt
    quality_convs = await data_collector.get_quality_conversations(min_score=0.8)
    
    # Chuẩn bị dữ liệu training
    train_data = []
    for conv in quality_convs:
        train_data.append({
            'input': conv['input'],
            'output': conv['output'],
            'context': conv['context']
        })
    
    # Fine-tune model
    if train_data:
        await model_manager.fine_tune(train_data)
        logging.info(f"Trained on {len(train_data)} quality conversations")
