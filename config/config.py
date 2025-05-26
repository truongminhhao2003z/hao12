from pathlib import Path
from typing import Dict, Any

class Config:
    # Đường dẫn thư mục
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"
    LOG_DIR = PROJECT_ROOT / "logs"
    CACHE_DIR = PROJECT_ROOT / "cache"
    MODEL_CACHE_DIR = PROJECT_ROOT / "model_cache"
    TOKENIZER_DIR = MODEL_CACHE_DIR / "tokenizer"  # Thêm dòng này
    MODEL_WEIGHTS_DIR = MODEL_CACHE_DIR / "model"   # Thêm dòng này

    # Cấu hình mặc định
    MAX_LENGTH = 128
    MIN_LENGTH = 10
    NUM_BEAMS = 5
    NO_REPEAT_NGRAM_SIZE = 2
    REPETITION_PENALTY = 1.5
    LENGTH_PENALTY = 1.0
    TOP_K = 50
    TOP_P = 0.9
    MAX_HISTORY = 10
    CACHE_TTL = 3600

    # Cấu hình Model
    MODEL_CONFIG = {
        'temperature': 0.7,
        'max_length': MAX_LENGTH,
        'min_length': MIN_LENGTH,
        'num_beams': NUM_BEAMS,
        'no_repeat_ngram_size': NO_REPEAT_NGRAM_SIZE,
        'repetition_penalty': REPETITION_PENALTY,
        'length_penalty': LENGTH_PENALTY,
        'top_k': TOP_K,
        'top_p': TOP_P
    }

    # Cấu hình cho DataCollector
    COLLECTOR_CONFIG = {
        'quality_thresholds': {
            'min_length': 10,  # Độ dài tối thiểu của câu
            'max_length': 1000,  # Độ dài tối đa
            'min_quality_score': 0.7,  # Điểm chất lượng tối thiểu
        },
        'min_quality_score': 0.8,  # Thêm dòng này
        'storage': {
            'max_conversations': 1000,
            'backup_interval': 24  # hours
        }
    }

    # Cấu hình Training
    TRAINING_CONFIG = {
        'batch_size': 4,
        'num_epochs': 5,
        'learning_rate': 2e-5,
        'warmup_steps': 100,
        'max_grad_norm': 1.0
    }
    
    @classmethod
    def create_directories(cls):
        """Tạo các thư mục cần thiết"""
        dirs = [
            cls.DATA_DIR / "conversations",
            cls.MODEL_DIR,
            cls.LOG_DIR,
            cls.CACHE_DIR,
            cls.MODEL_CACHE_DIR,
            cls.TOKENIZER_DIR,      # Thêm vào danh sách tạo thư mục
            cls.MODEL_WEIGHTS_DIR    # Thêm vào danh sách tạo thư mục
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(cls) -> Path:
        """Lấy đường dẫn đến model hiện tại"""
        return cls.MODEL_DIR / "best_model"

    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """Cấu hình logging"""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": str(cls.LOG_DIR / "chatbot.log"),
                    "level": "INFO",
                    "formatter": "standard",
                    "encoding": "utf-8"
                }
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": True
                }
            }
        }

# Khởi tạo thư mục khi import
Config.create_directories()