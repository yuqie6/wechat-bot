import time
import queue
import os
from wxauto import WeChat
from config import LISTEN_CONTACTS, GEMINI_API_KEY, GROUP_BOT_NAME
from gemini_handler import get_ai_response, clear_history, update_image_context, get_image_path_from_context
from logger import logger

# 创建一个线程安全的队列，用于在回调和主循环之间传递消息
task_queue = queue.Queue()
# 图片上下文管理已移至 gemini_handler.py
# 用于存储每个聊天窗口的最新图片信息（路径和时间戳）
# 结构: {'chat_name': {'path': 'path/to/image.png', 'timestamp': 1678886400}}
# last_image_context = {}
# IMAGE_CONTEXT_TTL = 3000 # 图片上下文的有效时间（秒），这里设置为50分钟

def message_callback(msg, chat):
    """
    这是在后台线程中运行的回调函数。
    它的作用是把收到的消息和关联的聊天窗口对象一起放入队列。
    """
    try:
        # --- 最终修复：根据日志，拍一拍事件的来源(attr)是'tickle'，而不是类型(type) ---
        # 我们现在监听所有非自己发送的消息，并在主循环中进行更精确的分类处理
        if msg.attr != 'self':
            task_queue.put((msg, chat))
    except Exception as e:
        logger.error(f"[回调错误] {e}")

def process_friend_requests(wx_instance):
    """
    定期检查并自动接受新的好友请求。
    """
    try:
        # 检查 wx_instance 是否具有 GetNewFriends 方法
        if hasattr(wx_instance, 'GetNewFriends'):
            logger.info("正在检查新的好友申请...")
            new_friends = wx_instance.GetNewFriends(acceptable=True)
            if not new_friends:
                logger.info("没有发现新的好友申请。")
                return

            logger.info(f"发现 {len(new_friends)} 条新的好友申请，正在处理...")
            for friend_request in new_friends:
                try:
                    # 为新好友创建一个备注
                    remark = f"自动添加_{friend_request.name}"
                    # 接受好友请求
                    friend_request.accept(remark=remark)
                    logger.info(f"已自动接受好友 '{friend_request.name}' 的申请，并设置备注为 '{remark}'。")
                    time.sleep(1) # 短暂延时，避免操作过快
                except Exception as e:
                    logger.error(f"处理好友 '{friend_request.name}' 的申请时失败: {e}")
        else:
            logger.warning("当前 wxauto 版本不支持 'GetNewFriends' 功能，已跳过好友申请检查。请安装 Plus 版 (wxautox) 以使用此功能。")
            
    except Exception as e:
        logger.error(f"检查好友申请时发生未知错误: {e}")


def main():
    """
    程序主入口，采用基于队列的被动监听模式。
    """
    opened_windows = set() # 用于记录已独立出来的群聊窗口
    last_friend_check_time = 0 # 上次检查好友申请的时间
    FRIEND_CHECK_INTERVAL = 300 # 每 300 秒（5分钟）检查一次

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
            # --- 定期任务：检查好友申请 ---
            current_time = time.time()
            if current_time - last_friend_check_time > FRIEND_CHECK_INTERVAL:
                process_friend_requests(wx)
                last_friend_check_time = current_time

            # --- 消息处理 ---
            # 从队列中获取任务，如果队列为空，会阻塞等待
            msg, chat = task_queue.get(timeout=60) # 设置超时以防永久阻塞

            # 获取聊天信息 (改用 chat 对象，更可靠)
            chat_info = chat.ChatInfo()
            chat_name = chat_info.get('chat_name', msg.sender) # 使用正确的 key 'chat_name'
            
            # 使用库内置方法判断是否为群聊
            is_group = chat_info.get('chat_type') == 'group'

            # --- 1. 分类预处理，提取内容 ---
            user_message = ""
            user_message = ""
            image_path = None
            text_response = None
            files_to_send = []

            # --- 最终修复：检查 msg.attr 而不是 msg.type ---
            if msg.attr == 'tickle':
                bot_name_in_tickle = GROUP_BOT_NAME.lstrip('@')
                logger.info(f"收到来自群聊的拍一拍消息。内容: '{msg.content}'，与机器人名 '{bot_name_in_tickle}' 匹配...")
                
                # 检查被拍的是否是机器人
                if bot_name_in_tickle in msg.content:
                    logger.info(f"匹配成功！将“拍一拍”事件作为AI输入。")
                    # 构造一个消息，让AI知道发生了什么
                    user_message = f"[{msg.sender} 拍了拍我]"
                    # should_process 将在下面的通用逻辑中被设置为 True
                else:
                    logger.info(f"匹配失败。这是一个与机器人无关的拍一拍消息，已忽略。")
                    continue # 如果是与机器人无关的拍一拍，直接跳过

            elif msg.type == 'image':
                try:
                    img_dir = "images"
                    if not os.path.exists(img_dir):
                        os.makedirs(img_dir)
                    logger.info(f"收到来自 [{msg.sender}] 的图片消息，正在下载...")
                    downloaded_path = msg.download(dir_path=img_dir)
                    logger.info(f"图片下载成功: {downloaded_path}")
                    
                    # 更新图片上下文，使用绝对路径
                    update_image_context(
                        chat_name=chat_name,
                        path=os.path.abspath(downloaded_path),
                        timestamp=time.time()
                    )
                    
                    # 收到图片后，引导用户输入指令
                    text_response = "图片收到！请告诉我需要对它做什么（例如：抠出图中的人像）。"
                    # logger.info(f"已为 [{chat_name}] 更新图片上下文，并发送引导语。") # 日志记录已移到handler中
                    
                except Exception as e:
                    logger.error(f"下载或处理图片上下文失败: {e}")
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
                user_message = msg.content.strip()


                # 使用 gemini_handler 中提供的函数来安全地获取图片路径
                image_path = get_image_path_from_context(chat_name)
                if image_path:
                    logger.info(f"找到与消息 '{user_message}' 关联的图片: {image_path}")
                else:
                    # 如果没有找到有效图片，image_path 会是 None，后续逻辑会正常处理
                    pass

            # --- 2. 应用群聊@规则 & 调用AI ---
            final_user_message = user_message
            should_process = False # 在这里初始化，确保它总是有值
            
            # 检查是否为特殊命令，并确定是否需要调用AI
            is_clear_command = (user_message == '清除历史记录')
            
            if is_group:
                at_name_with_symbol = f"@{GROUP_BOT_NAME}" if not GROUP_BOT_NAME.startswith('@') else GROUP_BOT_NAME
                if at_name_with_symbol in user_message:
                    stripped_message = user_message.replace(at_name_with_symbol, "").strip()
                    if stripped_message == '清除历史记录':
                        is_clear_command = True
                    else:
                        final_user_message = f"{msg.sender}: {stripped_message}"
                        should_process = True # 是@消息，且不是命令，需要AI处理
                elif not is_clear_command:
                    # 在群聊中，如果既不是@消息，也不是特殊命令，则忽略
                    continue
            else: # 私聊
                should_process = not is_clear_command

            # 根据标志执行操作
            if is_clear_command:
                logger.info(f"收到来自 [{chat_name}] 的清除历史记录命令。")
                if clear_history(chat_name):
                    text_response = "好的，我已经忘记我们之前聊过什么了。有什么新话题吗？"
                else:
                    text_response = "嗯...我好像还不认识你，没有找到我们的聊天记录。"
            
            if should_process and (final_user_message or image_path):
                logger.info(f"准备调用AI处理: '{final_user_message}' (图片: {'有' if image_path else '无'})")
                text_response, files_to_send = get_ai_response(
                    contact_name=chat_name,
                    user_message=final_user_message,
                    image_path=image_path,
                    is_group=is_group,
                    sender_name=msg.sender
                )
            elif not text_response and not is_clear_command:
                 logger.info(f"收到来自 [{msg.sender}] 的空消息或不需处理的消息，已忽略。")


            # --- 3. 发送最终回复 ---
            if text_response:
                logger.info(f"最终文本回复: {text_response}")
                try:
                    msg.quote(text_response)
                    logger.info("文本回复已发送。")
                except Exception as send_e:
                    logger.error(f"发送文本回复失败: {send_e}")
            
            if files_to_send:
                logger.info(f"准备发送 {len(files_to_send)} 个文件...")
                for file_path in files_to_send:
                    try:
                        chat.SendFiles(file_path)
                        logger.info(f"已发送文件: {file_path}")
                        time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"发送文件 {file_path} 失败: {e}")
        
        except queue.Empty:
            # 队列为空是正常情况，继续等待
            continue
        except Exception as e:
            logger.error(f"处理消息时发生未知错误: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()