from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path
import logging
from json.decoder import JSONDecodeError
from config.config import Config

class ConversationError(Exception):
    """Custom exception cho các lỗi liên quan đến conversation"""
    pass

class DataCollector:
    def __init__(self):
        self.data_dir = Config.DATA_DIR / "conversations"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
        # Thêm xử lý default config
        default_thresholds = {
            'min_length': 10,
            'max_length': 1000,
            'min_quality_score': 0.7
        }
        
        # Lấy config từ Config hoặc dùng default
        self.quality_thresholds = (
            Config.COLLECTOR_CONFIG.get('quality_thresholds', default_thresholds)
            if hasattr(Config, 'COLLECTOR_CONFIG')
            else default_thresholds
        )

    def _setup_logging(self):
        """Cấu hình logging"""
        log_file = self.data_dir / 'collector.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

    def _validate_inputs(self, user_text: str, bot_response: str) -> bool:
        """Kiểm tra input hợp lệ"""
        if not user_text or not bot_response:
            raise ConversationError("User text và bot response không được để trống")
        if len(user_text) > 1000:  # Giới hạn độ dài
            raise ConversationError("User text quá dài")
        return True

    async def save_conversation(
        self,
        user_text: str,
        bot_response: str,
        intent: str,
        context: Dict,
        quality_score: Optional[float] = None
    ) -> bool:
        """Lưu cuộc hội thoại vào file JSON"""
        try:
            # Validate inputs
            self._validate_inputs(user_text, bot_response)
            
            # Chuẩn bị tên file
            timestamp = datetime.now()
            filename = self.data_dir / f"conv_{timestamp.strftime('%Y%m')}.json"
            temp_file = filename.with_suffix('.tmp')
            backup_file = filename.with_suffix('.bak')
            
            # Chuẩn bị dữ liệu mới
            conversation = {
                "id": str(timestamp.timestamp()),
                "timestamp": timestamp.isoformat(),
                "input": user_text.strip(),
                "output": bot_response.strip(),
                "intent": intent,
                "context": context,
                "quality_score": quality_score
            }

            # Đọc dữ liệu cũ
            data = []
            if filename.exists():
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except JSONDecodeError:
                    logging.warning(f"File {filename} bị lỗi, thử đọc backup")
                    if backup_file.exists():
                        with open(backup_file, "r", encoding="utf-8") as f:
                            try:
                                data = json.load(f)
                            except JSONDecodeError:
                                logging.error("Backup file cũng bị lỗi")
                                data = []

            # Thêm conversation mới
            data.append(conversation)

            # Lưu vào file tạm
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Backup file cũ nếu cần
            if filename.exists():
                if backup_file.exists():
                    backup_file.unlink()  # Xóa backup cũ
                filename.rename(backup_file)

            # Đổi tên file tạm thành file chính
            temp_file.rename(filename)

            logging.info(f"✅ Đã lưu conversation {conversation['id']} vào {filename}")
            return True

        except ConversationError as e:
            logging.warning(f"❌ Lỗi validation: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"❌ Lỗi không mong muốn: {str(e)}")
            return False

    async def get_conversations(self, limit: int = 100) -> List[Dict]:
        """Đọc conversations gần đây nhất"""
        try:
            files = sorted(self.data_dir.glob("conv_*.json"), reverse=True)
            conversations = []
            
            for file in files:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    conversations.extend(data)
                if len(conversations) >= limit:
                    break
                    
            return conversations[:limit]
            
        except Exception as e:
            logging.error(f"Error reading conversations: {str(e)}")
            return []

    def evaluate_conversation_quality(self, user_text: str, bot_response: str, context: Dict) -> float:
        """Đánh giá chất lượng cuộc hội thoại để training
        
        Returns:
            float: Điểm chất lượng từ 0-1
        """
        score = 1.0
        
        # 1. Kiểm tra độ dài phù hợp
        if len(user_text) < self.quality_thresholds['min_length']:
            score *= 0.5
        if len(bot_response) < self.quality_thresholds['min_length']:
            score *= 0.5

        # 2. Đánh giá tính liên quan của context
        if context.get('intent') == 'other':
            score *= 0.7
        
        # 3. Kiểm tra tính hoàn chỉnh của câu trả lời
        if not bot_response.endswith(('.', '?', '!')):
            score *= 0.9
            
        # 4. Kiểm tra tỷ lệ từ trùng lặp
        user_words = set(user_text.lower().split())
        bot_words = set(bot_response.lower().split())
        word_overlap = len(user_words & bot_words) / len(user_words) if user_words else 0
        if word_overlap < 0.2:  # Quá ít từ liên quan
            score *= 0.8
        
        # 5. Đánh giá chất lượng ngữ cảnh
        context_quality = self._evaluate_context_quality(context)
        score *= context_quality

        return round(score, 2)

    def _evaluate_context_quality(self, context: Dict) -> float:
        """Đánh giá chất lượng của ngữ cảnh"""
        score = 1.0
        
        required_fields = ['time', 'intent', 'topic']
        for field in required_fields:
            if field not in context:
                score *= 0.9
        
        # Kiểm tra topic phù hợp
        if context.get('topic') == 'unknown':
            score *= 0.8
            
        return score

    async def get_quality_conversations(self, min_score: float = 0.7) -> List[Dict]:
        """Lấy các cuộc hội thoại có chất lượng tốt để training"""
        conversations = await self.get_conversations()
        return [
            conv for conv in conversations 
            if conv.get('quality_score', 0) >= min_score
        ]