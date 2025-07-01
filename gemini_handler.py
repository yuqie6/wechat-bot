import google.generativeai as genai
import json
import os
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
    genai.configure(api_key=GEMINI_API_KEY, transport="rest", client_options=opts)
else:
    genai.configure(api_key=GEMINI_API_KEY)

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
model = genai.GenerativeModel(
    'gemini-2.5-flash',
    system_instruction=SYSTEM_PROMPT
)

def get_ai_response(contact_name, user_message):
    """
    获取 AI 的回复。

    Args:
        contact_name (str): 联系人或群聊的名称.
        user_message (str): 用户发送的最新消息.

    Returns:
        str: AI 生成的回复内容.
    """
    # 如果是新的对话，尝试从文件加载历史记录
    if contact_name not in conversation_history:
        conversation_history[contact_name] = load_history(contact_name)

    # 将用户的新消息添加到历史记录中
    conversation_history[contact_name].append({'role': 'user', 'parts': [user_message]})

    try:
        # 开始一个聊天会话
        chat_session = model.start_chat(
            history=conversation_history[contact_name]
        )
        
        # 发送消息并获取回复
        response = chat_session.send_message(user_message)
        ai_response = response.text

        # 将 AI 的回复也添加到历史记录中
        conversation_history[contact_name].append({'role': 'model', 'parts': [ai_response]})
        
        # 保存更新后的历史记录
        save_history(contact_name)

        return ai_response

    except Exception as e:
        logger.error(f"调用 Gemini API 时出错: {e}")
        # 发生错误时，可以考虑从历史记录中移除最后一条用户消息
        conversation_history[contact_name].pop()
        return "抱歉，我现在有点忙，稍后再试吧。"
