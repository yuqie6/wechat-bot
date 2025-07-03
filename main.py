import asyncio
import os
import time
from wxauto import WeChat
import config
from logger import logger

# --- 从 config.py 加载配置 ---
GEMINI_API_KEY = getattr(config, 'GEMINI_API_KEY', 'YOUR_API_KEY')
LISTEN_CONTACTS = getattr(config, 'LISTEN_CONTACTS', ['文件传输助手'])
GROUP_BOT_NAME = getattr(config, 'GROUP_BOT_NAME', 'AI助手')
AUTO_ACCEPT_FRIENDS = getattr(config, 'AUTO_ACCEPT_FRIENDS', True)
FRIEND_REMARK_PREFIX = getattr(config, 'FRIEND_REMARK_PREFIX', 'AI添加_')
CLEAR_HISTORY_COMMAND = getattr(config, 'CLEAR_HISTORY_COMMAND', '清除历史记录')
IMAGE_DIR = getattr(config, 'IMAGE_DIR', 'images')
FRIEND_CHECK_INTERVAL = getattr(config, 'FRIEND_CHECK_INTERVAL', 300)
IMAGE_RECEIVED_PROMPT = getattr(config, 'IMAGE_RECEIVED_PROMPT', "图片收到！请告诉我需要对它做什么。")

# --- 导入重构后的异步AI处理函数 ---
from gemini_handler import get_ai_response_async, clear_history, update_image_context, get_image_path_from_context

# 创建一个异步任务队列
task_queue = asyncio.Queue()

def create_message_callback(loop: asyncio.AbstractEventLoop):
    """
    创建一个闭包，捕获事件循环，用于线程安全地将任务放入队列。
    """
    def message_callback(msg, chat):
        """
        这是在 wxauto 后台线程中运行的回调函数。
        它的作用是把收到的消息和关联的聊天窗口对象一起放入异步队列。
        """
        try:
            if msg.attr != 'self':
                # 使用 call_soon_threadsafe 从另一个线程安全地与 asyncio 事件循环交互
                loop.call_soon_threadsafe(task_queue.put_nowait, (msg, chat))
        except Exception as e:
            logger.error(f"[回调错误] {e}")
    return message_callback

def process_friend_requests(wx_instance):
    """
    检查并自动接受新的好友请求（同步函数）。
    """
    if not AUTO_ACCEPT_FRIENDS:
        return
        
    try:
        if not hasattr(wx_instance, 'GetNewFriends'):
            logger.warning("配置项 AUTO_ACCEPT_FRIENDS 已开启，但当前 wxauto 版本不支持 'GetNewFriends' 功能。请安装 Plus 版 (wxautox) 以使用此功能。")
            return

        logger.info("正在检查新的好友申请...")
        new_friends = wx_instance.GetNewFriends(acceptable=True)
        if not new_friends:
            logger.info("没有发现新的好友申请。")
            return

        logger.info(f"发现 {len(new_friends)} 条新的好友申请，正在处理...")
        for friend_request in new_friends:
            try:
                remark = f"{FRIEND_REMARK_PREFIX}{friend_request.name}"
                friend_request.accept(remark=remark)
                logger.info(f"已自动接受好友 '{friend_request.name}' 的申请，并设置备注为 '{remark}'。")
                time.sleep(1)
            except Exception as e:
                logger.error(f"处理好友 '{friend_request.name}' 的申请时失败: {e}")
            
    except Exception as e:
        logger.error(f"检查好友申请时发生未知错误: {e}")

async def friend_request_processor(wx_instance):
    """
    一个独立的异步任务，定期检查好友请求。
    """
    while True:
        try:
            process_friend_requests(wx_instance)
        except Exception as e:
            logger.error(f"好友请求处理器发生错误: {e}")
        await asyncio.sleep(FRIEND_CHECK_INTERVAL)

async def message_consumer(wx_instance):
    """
    异步消息消费者，从队列中获取消息并进行处理。
    """
    while True:
        try:
            msg, chat = await task_queue.get()

            chat_info = chat.ChatInfo()
            chat_name = chat_info.get('chat_name', msg.sender)
            is_group = chat_info.get('chat_type') == 'group'

            user_message = ""
            image_path = None
            text_response = None
            files_to_send = []

            if msg.attr == 'tickle':
                bot_name_in_tickle = GROUP_BOT_NAME.lstrip('@')
                if bot_name_in_tickle in msg.content:
                    user_message = f"[{msg.sender} 拍了拍我]"
                else:
                    continue
            elif msg.type == 'image':
                try:
                    if not os.path.exists(IMAGE_DIR):
                        os.makedirs(IMAGE_DIR)
                    downloaded_path = msg.download(dir_path=IMAGE_DIR)
                    update_image_context(
                        chat_name=chat_name,
                        path=os.path.abspath(downloaded_path),
                        timestamp=time.time()
                    )
                    text_response = IMAGE_RECEIVED_PROMPT
                except Exception as e:
                    logger.error(f"下载或处理图片上下文失败: {e}")
                    continue
            elif msg.type == 'voice':
                try:
                    user_message = msg.to_text()
                    if not user_message:
                        continue
                except Exception as e:
                    logger.error(f"语音转文字失败: {e}")
                    continue
            elif msg.type == 'text':
                user_message = msg.content.strip()
                image_path = get_image_path_from_context(chat_name)

            final_user_message = user_message
            should_process = False
            is_clear_command = (user_message == CLEAR_HISTORY_COMMAND)

            if is_group:
                at_name_with_symbol = f"@{GROUP_BOT_NAME}" if not GROUP_BOT_NAME.startswith('@') else GROUP_BOT_NAME
                if at_name_with_symbol in user_message:
                    stripped_message = user_message.replace(at_name_with_symbol, "").strip()
                    if stripped_message == CLEAR_HISTORY_COMMAND:
                        is_clear_command = True
                    else:
                        final_user_message = f"{msg.sender}: {stripped_message}"
                        should_process = True
                elif not is_clear_command:
                    continue
            else:
                should_process = not is_clear_command

            if is_clear_command:
                if clear_history(chat_name):
                    text_response = "好的，我已经忘记我们之前聊过什么了。有什么新话题吗？"
                else:
                    text_response = "嗯...我好像还不认识你，没有找到我们的聊天记录。"
            
            if should_process and (final_user_message or image_path):
                logger.info(f"准备调用AI处理: '{final_user_message}' (图片: {'有' if image_path else '无'})")
                # 调用异步AI处理函数
                text_response, files_to_send = await get_ai_response_async(
                    contact_name=chat_name,
                    user_message=final_user_message,
                    image_path=image_path,
                    is_group=is_group,
                    sender_name=msg.sender
                )
            elif not text_response and not is_clear_command:
                logger.info(f"收到来自 [{msg.sender}] 的空消息或不需处理的消息，已忽略。")

            if text_response:
                try:
                    msg.quote(text_response)
                    logger.info(f"向 [{chat_name}] 发送文本回复成功。")
                except Exception as send_e:
                    logger.error(f"发送文本回复失败: {send_e}")
            
            if files_to_send:
                for file_path in files_to_send:
                    try:
                        chat.SendFiles(file_path)
                        logger.info(f"向 [{chat_name}] 发送文件成功: {file_path}")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"发送文件 {file_path} 失败: {e}")
            
            task_queue.task_done()

        except Exception as e:
            logger.error(f"处理消息时发生未知错误: {e}", exc_info=True)
            # 发生错误后短暂休息，避免快速失败循环
            await asyncio.sleep(1)

async def main():
    """
    程序主入口，异步模式。
    """
    logger.info("--- 微信 AI 机器人启动中 (异步模式) ---")

    if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_API_KEY':
        logger.error("错误: 请在 .env 文件或 config.py 中配置您的 GEMINI_API_KEY。")
        return

    logger.info("正在初始化微信实例...")
    try:
        wx = WeChat()
        wx.Show()
        logger.info("微信实例获取成功。")
    except Exception as e:
        logger.error(f"获取微信实例失败: {e}")
        return

    # 获取当前事件循环，并创建线程安全的回调
    loop = asyncio.get_running_loop()
    callback = create_message_callback(loop)

    if not LISTEN_CONTACTS:
        logger.warning("监听列表为空，机器人不会对任何消息做出反应。")
    else:
        logger.info(f"开始为 {len(LISTEN_CONTACTS)} 个联系人添加监听...")
        for contact in LISTEN_CONTACTS:
            try:
                wx.AddListenChat(nickname=contact, callback=callback)
                logger.info(f"  - 已成功添加对 [{contact}] 的监听。")
            except Exception as e:
                logger.error(f"  - 添加对 [{contact}] 的监听失败: {e}")
    
    wx.StartListening()
    logger.info("--- 机器人已成功启动，正在等待消息... ---")

    # 创建并启动后台任务
    consumer_task = asyncio.create_task(message_consumer(wx))
    friend_checker_task = asyncio.create_task(friend_request_processor(wx))

    # 等待任务完成（实际上是永久运行）
    await asyncio.gather(consumer_task, friend_checker_task)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("检测到 Ctrl+C，机器人正在关闭...")
    except Exception as e:
        logger.critical(f"主程序发生致命错误: {e}", exc_info=True)