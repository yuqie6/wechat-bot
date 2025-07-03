import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# --- 类型转换辅助函数 ---
def get_bool(key, default_value):
    val = os.getenv(key, str(default_value))
    return val.lower() in ('true', '1', 't')

def get_int(key, default_value):
    return int(os.getenv(key, default_value))

def get_list(key, default_value):
    val = os.getenv(key, default_value)
    return [item.strip() for item in val.split(',') if item.strip()]

# =================================================================
# ================== Gemini AI Assistant Config ===================
# =================================================================
#
#  所有配置项均从 .env 文件加载，此处仅定义和转换类型。
#  如需修改配置，请编辑项目根目录下的 .env 文件。
#
# =================================================================


# ---------------------- API & Model Config -----------------------
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_API_KEY')
GEMINI_BASE_URL = os.getenv('GEMINI_BASE_URL')
ENABLE_GOOGLE_SEARCH = get_bool('ENABLE_GOOGLE_SEARCH', True) # 新增：联网搜索功能开关

# [重要] AI 初始化提示词 (系统指令)
def _load_system_prompt(primary_path='prompt.txt', fallback_path='prompt.txt.template'):
    """
    优先从 primary_path 加载系统提示词。
    如果 primary_path 不存在，则尝试从 fallback_path 加载。
    如果两个文件都找不到，则返回一个硬编码的默认提示。
    """
    try:
        # 优先尝试主文件
        with open(primary_path, 'r', encoding='utf-8') as f:
            print(f"成功从 '{primary_path}' 加载系统提示。")
            return f.read()
    except FileNotFoundError:
        # 如果主文件不存在，尝试备用模板文件
        print(f"提示: 未找到 '{primary_path}'。正在尝试从模板文件 '{fallback_path}' 加载...")
        try:
            with open(fallback_path, 'r', encoding='utf-8') as f:
                print(f"成功从 '{fallback_path}' 加载系统提示。")
                return f.read()
        except FileNotFoundError:
            # 如果两个文件都找不到
            print(f"警告: '{primary_path}' 和 '{fallback_path}' 均未找到。将使用默认的系统提示。")
            return "你是一个乐于助人的AI助手。"

SYSTEM_PROMPT = _load_system_prompt()


# ---------------------- Robot Behavior Config --------------------
LISTEN_CONTACTS = get_list('LISTEN_CONTACTS', '文件传输助手')
GROUP_BOT_NAME = os.getenv('GROUP_BOT_NAME', 'AI助手')
AUTO_ACCEPT_FRIENDS = get_bool('AUTO_ACCEPT_FRIENDS', True)
FRIEND_REMARK_PREFIX = os.getenv('FRIEND_REMARK_PREFIX', 'AI添加_') 
CLEAR_HISTORY_COMMAND = os.getenv('CLEAR_HISTORY_COMMAND', '清除历史记录')
MAX_HISTORY_TURNS = get_int('MAX_HISTORY_TURNS', 10) # 不设置默认保留最近10轮对话作为上下文


# ---------------------- Image Processing Config ------------------
IMAGE_DIR = os.getenv('IMAGE_DIR', 'images')
IMAGE_CONTEXT_TTL = get_int('IMAGE_CONTEXT_TTL', 3000)
IMAGE_RECEIVED_PROMPT = os.getenv('IMAGE_RECEIVED_PROMPT', "图片收到！请告诉我需要对它做什么。")


# ---------------------- System & Data Config ---------------------
HISTORY_DIR = os.getenv('HISTORY_DIR', 'history')
LOG_DIR = os.getenv('LOG_DIR', 'logs')
LOG_FILE_MAX_SIZE = get_int('LOG_FILE_MAX_SIZE', 10)
LOG_FILE_BACKUP_COUNT = get_int('LOG_FILE_BACKUP_COUNT', 5)
FRIEND_CHECK_INTERVAL = get_int('FRIEND_CHECK_INTERVAL', 300)
