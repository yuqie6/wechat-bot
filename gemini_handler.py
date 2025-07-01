import json
import os
import PIL.Image
from google.generativeai.client import configure
from google.generativeai.generative_models import GenerativeModel
from google.api_core import client_options
from config import GEMINI_API_KEY, GEMINI_BASE_URL, SYSTEM_PROMPT, HISTORY_DIR
from logger import logger

# 确保历史记录目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 配置 API 客户端
if GEMINI_BASE_URL:
    # 清理用户提供的 base_url，移除末尾的斜杠和可能的版本路径，以防止重复
    cleaned_url = GEMINI_BASE_URL.strip().rstrip('/')
    if cleaned_url.endswith('/v1beta'):
        cleaned_url = cleaned_url[:-len('/v1beta')]
        
    opts = client_options.ClientOptions(api_endpoint=cleaned_url)
    configure(api_key=GEMINI_API_KEY, transport="rest", client_options=opts)
else:
    configure(api_key=GEMINI_API_KEY)

# 对话历史记录
# 结构: {'联系人/群聊名': [message, message, ...]}
conversation_history = {}

def save_history(contact_name):
    """将指定联系人的对话历史保存到文件"""
    if contact_name in conversation_history:
        # 文件名不允许包含特殊字符，进行简单替换
        safe_filename = contact_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        file_path = os.path.join(HISTORY_DIR, f"{safe_filename}.json")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_history[contact_name], f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存历史记录失败 ({contact_name}): {e}")

def load_history(contact_name):
    """从文件加载指定联系人的对话历史"""
    safe_filename = contact_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    file_path = os.path.join(HISTORY_DIR, f"{safe_filename}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载历史记录失败 ({contact_name}): {e}")
    return []

# 初始化 Gemini Pro 模型
model = GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=SYSTEM_PROMPT
)

def get_ai_response(contact_name, user_message, image_path=None, is_group=False, sender_name=None):
    """
    获取 AI 的回复，支持文本、图片，并能感知群聊中的不同用户。

    Args:
        contact_name (str): 聊天窗口的名称 (群名或好友名).
        user_message (str): 用户发送的最新消息 (在群聊中已包含发言人前缀).
        image_path (str, optional): 附带的图片路径. Defaults to None.
        is_group (bool, optional): 是否为群聊. Defaults to False.
        sender_name (str, optional): 在群聊中的消息发送者昵称. Defaults to None.

    Returns:
        str: AI 生成的回复内容.
    """
    try:
        # --- 动态构建系统指令 ---
        current_system_prompt = SYSTEM_PROMPT
        if is_group and sender_name:
            # 为群聊场景动态添加指令，告知AI当前发言人是谁
            current_system_prompt += f"\n\n请注意：你正在一个群聊中，当前向你提问的用户是“{sender_name}”。请结合上下文，并以对“{sender_name}”说话的口吻进行回复。"
        
        # 每次都创建一个新的模型实例以应用最新的系统指令
        # 注意：这可能会轻微增加延迟，但在需要动态指令时是必要的
        dynamic_model = GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=current_system_prompt
        )

        # --- 处理图片消息 ---
        if image_path:
            logger.info(f"正在准备发送图片 {image_path} 和文字 '{user_message}' 给AI。")
            img = PIL.Image.open(image_path)
            # 使用动态创建的模型实例
            response = dynamic_model.generate_content([user_message, img])
            return response.text

        # --- 处理纯文本消息 ---
        if contact_name not in conversation_history:
            conversation_history[contact_name] = load_history(contact_name)

        # 使用动态创建的模型实例开始聊天
        chat_session = dynamic_model.start_chat(
            history=conversation_history[contact_name]
        )
        
        response = chat_session.send_message(user_message)
        ai_response = response.text

        # 手动将gemini返回的复杂历史对象转换为可序列化的字典列表
        serializable_history = []
        for content in chat_session.history:
            # 确保只处理有文本部分的消息
            parts_text = [part.text for part in content.parts if hasattr(part, 'text')]
            if parts_text:
                serializable_history.append({'role': content.role, 'parts': parts_text})

        # 使用转换后的安全格式更新历史记录
        conversation_history[contact_name] = serializable_history
        save_history(contact_name)

        return ai_response

    except Exception as e:
        logger.error(f"调用 Gemini API 时出错: {e}")
        if not image_path and contact_name in conversation_history and conversation_history[contact_name]:
            conversation_history[contact_name].pop()
        return "抱歉，我现在有点忙，稍后再试吧。"
