import logging
import os
from logging.handlers import TimedRotatingFileHandler

# 日志文件存放的目录
LOG_DIR = 'logs'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 创建一个 logger
logger = logging.getLogger('wechat_bot')
logger.setLevel(logging.INFO)

# 创建一个 handler，用于写入日志文件
# TimedRotatingFileHandler 会按时间自动分割日志文件
# when='D' 表示每天分割一次，backupCount=7 表示保留最近7天的日志
handler = TimedRotatingFileHandler(
    os.path.join(LOG_DIR, 'bot.log'),
    when='D',
    interval=1,
    backupCount=30,
    encoding='utf-8'
)
handler.setLevel(logging.INFO)

# 创建一个 handler，用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义 handler 的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 给 logger 添加 handler
if not logger.handlers:
    logger.addHandler(handler)
    logger.addHandler(console_handler)