import time
import queue
from wxauto import WeChat
from config import LISTEN_CONTACTS, GEMINI_API_KEY, GROUP_BOT_NAME
from gemini_handler import get_ai_response
from logger import logger

# 创建一个线程安全的队列，用于在回调和主循环之间传递消息
task_queue = queue.Queue()

def message_callback(msg, chat):
    """
    这是在后台线程中运行的回调函数。
    它的作用是把收到的消息放入队列，然后立即返回。
    chat 参数是库要求的，但我们在这里用不到它。
    """
    try:
        # 过滤掉非文本和自己发送的消息
        if msg.type == 'text' and msg.attr != 'self':
            # 将消息对象直接放入队列
            task_queue.put(msg)
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
            msg = task_queue.get(timeout=60) # 设置超时以防永久阻塞

            # 获取聊天信息
            chat_info = msg.chat_info()
            chat_name = chat_info.get('name', msg.sender)
            is_group = chat_info.get('type') == 'group'

            # 如果是群聊，并且是第一次处理该群聊的消息，则双击独立出窗口
            if is_group and chat_name not in opened_windows:
                try:
                    # 遍历会话列表找到对应的群聊并双击
                    sessions = wx.GetSession()
                    for session in sessions:
                        if session.name == chat_name:
                            session.double_click()
                            opened_windows.add(chat_name)
                            logger.info(f"已成功将群聊 [{chat_name}] 窗口独立出来。")
                            time.sleep(0.5) # 等待窗口弹出
                            break
                except Exception as e:
                    logger.error(f"尝试独立群聊 [{chat_name}] 窗口失败: {e}")
            
            # 提取消息内容，并为群聊做预处理
            user_message = msg.content
            
            # 群聊处理逻辑：必须 @机器人才回复
            if is_group:
                at_name = f"@{GROUP_BOT_NAME}"
                if at_name not in user_message:
                    continue # 如果没有@机器人，则忽略此消息
                
                # 移除@信息，得到干净的用户问题
                user_message = user_message.replace(at_name, "").strip()

            logger.info(f"在 [{chat_name}] 中收到来自 [{msg.sender}] 的有效消息: {user_message}")

            # 调用 AI 获取回复
            ai_response = get_ai_response(chat_name, user_message)

            # 发送回复
            if ai_response:
                logger.info(f"AI 生成的回复: {ai_response}")
                try:
                    # 直接使用消息对象进行引用回复
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