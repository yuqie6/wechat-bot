import time
import queue
import os
from wxauto import WeChat
from config import LISTEN_CONTACTS, GEMINI_API_KEY, GROUP_BOT_NAME
from gemini_handler import get_ai_response
from logger import logger

# 创建一个线程安全的队列，用于在回调和主循环之间传递消息
task_queue = queue.Queue()

def message_callback(msg, chat):
    """
    这是在后台线程中运行的回调函数。
    它的作用是把收到的消息和关联的聊天窗口对象一起放入队列。
    """
    try:
        # 允许处理文本、语音、图片和拍一拍消息，并过滤掉自己发送的
        allowed_types = ['text', 'voice', 'tickle', 'image']
        if msg.type in allowed_types and msg.attr != 'self':
            # 将消息和聊天窗口对象的元组放入队列
            task_queue.put((msg, chat))
    except Exception as e:
        logger.error(f"[回调错误] {e}")

def main():
    """
    程序主入口，采用基于队列的被动监听模式。
    """
    opened_windows = set() # 用于记录已独立出来的群聊窗口
    logger.info("--- 微信 AI 机器人启动中 (被动监听模式) ---")

    if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_API_KEY':
        logger.error("错误: 请在 config.py 文件中配置您的 GEMINI_API_KEY。")
        return

    logger.info("正在初始化微信实例...")
    try:
        wx = WeChat()
        wx.Show()
        logger.info("微信实例获取成功。")
    except Exception as e:
        logger.error(f"获取微信实例失败: {e}")
        return

    # 为指定联系人添加监听
    if not LISTEN_CONTACTS:
        logger.warning("监听列表为空，机器人不会对任何消息做出反应。")
    else:
        logger.info(f"开始为 {len(LISTEN_CONTACTS)} 个联系人添加监听...")
        for contact in LISTEN_CONTACTS:
            try:
                wx.AddListenChat(nickname=contact, callback=message_callback)
                logger.info(f"  - 已成功添加对 [{contact}] 的监听。")
            except Exception as e:
                logger.error(f"  - 添加对 [{contact}] 的监听失败: {e}")
    
    # 启动监听（非阻塞）
    wx.StartListening()
    logger.info("--- 机器人已成功启动，正在被动等待消息... ---")
    logger.info("--- 这个模式不会抢占您的鼠标，可以正常使用电脑 ---")

    # 主循环（消费者）
    while True:
        try:
            # 从队列中获取任务，如果队列为空，会阻塞等待
            msg, chat = task_queue.get(timeout=60) # 设置超时以防永久阻塞

            # 获取聊天信息 (改用 chat 对象，更可靠)
            chat_info = chat.ChatInfo()
            chat_name = chat_info.get('chat_name', msg.sender) # 使用正确的 key 'chat_name'
            
            # 使用库内置方法判断是否为群聊
            is_group = chat_info.get('chat_type') == 'group'

            # --- 1. 分类预处理，提取内容 ---
            user_message = ""
            image_path = None
            ai_response = None

            if msg.type == 'tickle':
                bot_name_in_tickle = GROUP_BOT_NAME.lstrip('@')
                if bot_name_in_tickle in msg.content:
                    logger.info(f"被 [{msg.sender}] 拍了拍，发送一个俏皮的回复。")
                    msg.quote("（づ￣3￣）づ╭❤～ 别拍啦，再拍就坏掉啦！")
                else:
                    logger.info(f"收到了一个与机器人无关的拍一拍消息，已忽略。")
                continue

            elif msg.type == 'image':
                try:
                    img_dir = "images"
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    logger.info(f"收到来自 [{msg.sender}] 的图片消息，正在下载...")
                    image_path = msg.download(dir_path=img_dir)
                    user_message = msg.content  # 图片附带的文字
                    logger.info(f"图片下载成功: {image_path}")
                except Exception as e:
                    logger.error(f"下载图片失败: {e}")
                    continue
            
            elif msg.type == 'voice':
                try:
                    logger.info(f"收到来自 [{msg.sender}] 的语音消息，正在转为文字...")
                    user_message = msg.to_text()
                    if not user_message:
                        logger.warning("语音消息转换为空文本，已忽略。")
                        continue
                    logger.info(f"语音转换结果: '{user_message}'")
                except Exception as e:
                    logger.error(f"语音转文字失败: {e}")
                    continue
            
            elif msg.type == 'text':
                user_message = msg.content

            # --- 2. 应用群聊@规则 ---
            if is_group:
                at_name_with_symbol = f"@{GROUP_BOT_NAME}" if not GROUP_BOT_NAME.startswith('@') else GROUP_BOT_NAME
                if at_name_with_symbol not in user_message:
                    continue # 在群聊中，如果没有@机器人，则忽略
                user_message = user_message.replace(at_name_with_symbol, "").strip()

            # --- 3. 根据内容调用AI或进行其他处理 ---
            if image_path:
                # 如果用户没有附带文字，我们使用一个默认提示
                prompt = user_message if user_message else "请详细描述这张图片，分析其中的内容和场景。"
                logger.info(f"准备调用AI分析图片: {image_path}, 附带消息: '{prompt}'")
                ai_response = get_ai_response(chat_name, prompt, image_path=image_path, is_group=is_group, sender_name=msg.sender)
            
            elif user_message:
                # 为群聊消息加上发送者前缀，以便在历史记录中区分
                final_user_message = f"{msg.sender}: {user_message}" if is_group else user_message
                logger.info(f"在 [{chat_name}] 中收到来自 [{msg.sender}] 的有效消息: {user_message}")
                ai_response = get_ai_response(chat_name, final_user_message, is_group=is_group, sender_name=msg.sender)
            
            else: # 处理空消息（如仅@）
                if is_group:
                    ai_response = "你好，我在！有什么可以帮你的吗？"
                    logger.info(f"收到来自 [{msg.sender}] 的空@提及，已发送预设回复。")
                else:
                    logger.info(f"收到来自 [{msg.sender}] 的空消息，已忽略。")

            # --- 4. 发送最终回复 ---
            if ai_response:
                logger.info(f"最终回复内容: {ai_response}")
                try:
                    msg.quote(ai_response)
                    logger.info("回复已发送。")
                except Exception as send_e:
                    logger.error(f"发送回复失败: {send_e}")
        
        except queue.Empty:
            # 队列为空是正常情况，继续等待
            continue
        except Exception as e:
            logger.error(f"处理消息时发生未知错误: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()