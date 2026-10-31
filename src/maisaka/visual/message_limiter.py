"""多模态消息图片数量限制工具；保留已有导入入口和普通图片配额语义。"""

from src.llm_models.image_budget import IMAGE_LIMIT_PLACEHOLDER, limit_latest_images_in_messages

__all__ = ["IMAGE_LIMIT_PLACEHOLDER", "limit_latest_images_in_messages"]
