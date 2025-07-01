import json
import os
import PIL.Image
import io
import base64
import binascii
import numpy as np
import pickle
from typing import Optional, Any, List, Dict
from PIL import Image, ImageDraw
import time
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_BASE_URL, SYSTEM_PROMPT, HISTORY_DIR
from logger import logger

# --- 全局设置 ---

# 确保历史记录目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 配置 API 客户端
# 适配 google-genai SDK，并处理自定义端点
if GEMINI_BASE_URL:
    cleaned_url = GEMINI_BASE_URL.strip().rstrip('/')
    logger.info(f"正在使用自定义端点: {cleaned_url}")
    # 通过 http_options 设置 base_url，将请求路由到自定义端点
    http_opts = types.HttpOptions(base_url= cleaned_url)
    # 同时传递 api_key，让 SDK 负责标准认证流程
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
else:
    # 谷歌官方端点的标准初始化
    client = genai.Client(api_key=GEMINI_API_KEY)

# --- 图片上下文管理 ---
# 结构: {'chat_name': {'path': 'path/to/image.png', 'timestamp': 1678886400}}
last_image_context = {}
IMAGE_CONTEXT_TTL = 3000 # 图片上下文的有效时间（秒），这里设置为50分钟

def update_image_context(chat_name: str, path: str, timestamp: float):
    """由 main.py 调用，用于更新指定聊天的最新图片信息。"""
    global last_image_context
    last_image_context[chat_name] = {'path': path, 'timestamp': timestamp}
    logger.info(f"已为 [{chat_name}] 更新图片上下文。")

def get_image_path_from_context(chat_name: str) -> Optional[str]:
    """由 main.py 调用，检查并获取有效的图片路径。"""
    context = last_image_context.get(chat_name)
    if context and time.time() - context.get('timestamp', 0) < IMAGE_CONTEXT_TTL:
        return context.get('path')
    elif context:
        # 如果上下文存在但已过期，则清理
        del last_image_context[chat_name]
        logger.info(f"'{chat_name}' 的图片上下文已过期并被清除。")
    return None

# --- 对话历史记录 ---
SESSIONS_FILE = os.path.join(HISTORY_DIR, 'sessions.pkl')

def _load_sessions() -> Dict[str, Any]:
    """从文件加载对话会话。"""
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'rb') as f:
                logger.info(f"从 '{SESSIONS_FILE}' 加载历史记录...")
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError) as e:
            logger.error(f"加载历史记录失败: {e}。将创建新的历史记录文件。")
    return {}

def _save_sessions(sessions: Dict[str, Any]):
    """将对话会话保存到文件。"""
    try:
        with open(SESSIONS_FILE, 'wb') as f:
            pickle.dump(sessions, f)
            logger.info(f"历史记录已成功保存到 '{SESSIONS_FILE}'。")
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")

# 初始化时加载对话会话
conversation_sessions = _load_sessions()

# --- 工具定义 ---

def _parse_json_from_gemini(json_output: str) -> str:
    """从模型的Markdown代码块中解析出纯JSON字符串。"""
    if "```json" in json_output:
        json_output = json_output.split("```json")[1].split("```")[0]
    return json_output.strip()

def segment_image(chat_name: str, user_prompt: str) -> dict:
    """
    当用户想要从图片中提取、分割或抠出某些物体时，调用此工具。
    工具会自动查找当前聊天会话中的最新图片进行处理。

    Args:
        chat_name (str): 当前聊天的名称，用于查找图片上下文。
        user_prompt (str): 用户的具体指令，描述了想要分割出的对象。例如：“把图里的猫抠出来”。

    Returns:
        dict: 一个包含操作结果的字典。
    """
    output_dir = "segmentation_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 从上下文中获取图片路径
    context = last_image_context.get(chat_name)
    if not context or time.time() - context.get('timestamp', 0) > IMAGE_CONTEXT_TTL:
        logger.warning(f"工具 'segment_image' 无法为 '{chat_name}' 找到有效的图片上下文。")
        return {'status': 'failure', 'message': '我需要你先发一张图片，然后我才能处理。'}
    
    image_path = context['path']

    try:
        logger.info(f"工具 'segment_image' 已被调用，聊天: {chat_name}, 图片: {image_path}, 指令: {user_prompt}")
        img = PIL.Image.open(image_path)
        img.thumbnail((1024, 1024), PIL.Image.Resampling.LANCZOS)

        prompt = f"""
        根据用户的指令 "{user_prompt}"，对图像中的对象进行分割。
        输出一个 JSON 列表，每个条目包含 "box_2d", "mask", 和 "label"。
        """
        # 最终修复：根据官方文档，使用 automatic_function_calling 参数来彻底禁用工具内的嵌套工具调用
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', # 或者 gemini-2.0-pro，根据可用性选择
            contents=[prompt, img],
            config=config
        )

        if not response.text:
            logger.error("模型在 segment_image 内部调用时返回了空文本。")
            return {'status': 'failure', 'message': '抱歉，模型没有返回有效的 JSON 数据。'}

        logger.info(f"模型在 segment_image 中返回的原始响应: {response.text}")
        
        parsed_json_str = _parse_json_from_gemini(response.text)
        logger.info(f"经过解析后，准备加载的 JSON 字符串: {parsed_json_str}")
        
        items = json.loads(parsed_json_str)
        if not items:
            return {'status': 'failure', 'message': '抱歉，我没能在图片中识别出任何可分割的对象。'}

        generated_files = []
        base_img_rgba = img.convert('RGBA')

        for i, item in enumerate(items):
            box = item.get("box_2d")
            png_b64 = item.get("mask")
            label = item.get("label", "unknown")

            if not all([box, png_b64, label]):
                continue
            
            try:
                mask_data = base64.b64decode(png_b64.removeprefix("data:image/png;base64,"))
                mask_img = Image.open(io.BytesIO(mask_data))
            except (binascii.Error, ValueError) as e:
                logger.warning(f"跳过一个无效的 Base64 mask 数据 (来自模型): {e}")
                continue

            y0, x0, y1, x1 = [int(c / 1000 * s) for c, s in zip(box, [img.size[1], img.size[0], img.size[1], img.size[0]])]
            if y0 >= y1 or x0 >= x1:
                continue

            mask_resized = mask_img.resize((x1 - x0, y1 - y0), PIL.Image.Resampling.BILINEAR)
            mask_array = np.array(mask_resized)

            # --- 改进后的抠图逻辑 ---
            # 1. 将原始图片转换为RGBA（如果它还不是）
            #    base_img_rgba 已经从外部作用域获得
            
            # 2. 创建一个与原图大小相同的、完全透明的画布
            cutout_image = Image.new('RGBA', base_img_rgba.size, (0, 0, 0, 0))

            # 3. 将模型返回的（可能很小的）mask图缩放到目标物体在原图中的实际尺寸
            mask_resized = mask_img.resize((x1 - x0, y1 - y0), PIL.Image.Resampling.BILINEAR)

            # 4. 创建一个与原图大小相同的、黑色的完整蒙版
            full_mask = Image.new('L', base_img_rgba.size, 0)
            # 5. 将缩放后的物体蒙版粘贴到完整蒙版的正确位置
            full_mask.paste(mask_resized, (x0, y0))

            # 6. 使用这个蒙版，从原始图片中“抠出”物体，粘贴到我们透明的画布上
            cutout_image = Image.composite(base_img_rgba, cutout_image, full_mask)

            # 7. 保存这张背景透明的抠图结果
            safe_label = "".join(c for c in label if c.isalnum())
            output_filename = f"{safe_label}_{i}.png" # 必须是PNG格式以支持透明度
            output_path = os.path.abspath(os.path.join(output_dir, output_filename))
            cutout_image.save(output_path)
            generated_files.append(output_path)

        if not generated_files:
            return {'status': 'failure', 'message': '我尝试处理了，但未能成功生成任何分割图片。'}

        return {
            'status': 'success',
            'message': f"成功处理并生成了 {len(generated_files)} 张图片。",
            'generated_files': generated_files
        }

    except Exception as e:
        logger.error(f"工具 'segment_image' 执行失败: {e}", exc_info=True)
        return {'status': 'error', 'message': f"抱歉，处理图片时遇到了一个内部错误: {e}"}

# --- 主逻辑 ---

def get_ai_response(contact_name: str, user_message: str, image_path: Optional[str] = None, is_group: bool = False, sender_name: Optional[str] = None) -> tuple[str, list]:
    """
    使用无状态的 generate_content API 获取AI回复，并手动管理对话历史。
    """
    try:
        # 获取或创建聊天历史
        if contact_name not in conversation_sessions or not isinstance(conversation_sessions.get(contact_name), list):
            logger.info(f"为 '{contact_name}' 创建新的聊天历史或重置无效的历史记录。")
            conversation_sessions[contact_name] = []
        
        history = conversation_sessions[contact_name]

        # 准备发送给模型的内容
        final_user_message = user_message
        if is_group and sender_name:
            final_user_message = f"请注意：你正在一个群聊中，当前向你提问的用户是“{sender_name}”。请结合上下文，并以对“{sender_name}”说话的口吻进行回复。\n\n用户的原始问题是：{user_message}"
        
        prompt_parts: List[Any] = []
        if image_path:
            logger.info(f"消息中包含图片: {image_path}")
            try:
                # 当有图片时，不仅要将图片对象加入，还要在文本中明确告知模型图片的路径
                img_for_prompt = PIL.Image.open(image_path)
                prompt_parts.append(img_for_prompt)
                # 移除硬编码的路径指令
                # final_user_message += f"\n\n[重要系统指令：用户提供的图片本地路径是 '{image_path}']"
            except Exception as e:
                logger.error(f"无法打开图片文件 {image_path}: {e}")
                return f"抱歉，我无法处理您发送的图片：{e}", []

        prompt_parts.append(final_user_message if final_user_message else " ")

        # 将历史记录和当前消息合并
        full_contents = history + prompt_parts
        
        logger.info(f"向 Gemini 发送内容: {final_user_message}, 图片路径: {image_path}, 历史记录条数: {len(history)}")
        
        # --- 半自动工具调用流程 ---

        # 1. 第一次调用：让模型决定是否要使用工具
        logger.info("第一步：请求模型进行工具调用决策...")
        first_call_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[segment_image],
            # 禁用自动调用，我们自己来控制
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_contents,
            config=first_call_config
        )
        
        logger.debug(f"第一次调用响应: {response}")

        final_text = ""
        generated_files = []
        model_response_content = None # 用于历史记录

        # 2. 检查模型是否请求了工具调用
        if response.function_calls:
            logger.info("模型请求调用工具，进入手动处理流程...")
            # 保存模型的工具调用请求，用于历史记录
            if response.candidates:
                model_response_content = response.candidates[0].content
            
            # 准备一个列表来存放工具的执行结果
            tool_response_parts = []
            
            for func_call in response.function_calls:
                if func_call.name == 'segment_image' and func_call.args:
                    # 提取模型建议的参数
                    user_prompt_from_model = func_call.args.get('user_prompt', '')
                    
                    # 3. 手动调用工具，但使用我们自己维护的、正确的 chat_name
                    logger.info(f"手动调用 'segment_image'，使用正确的 chat_name: '{contact_name}'")
                    tool_result = segment_image(
                        chat_name=contact_name,
                        user_prompt=user_prompt_from_model
                    )
                    
                    # 从成功的结果中提取文件
                    if tool_result.get('status') == 'success':
                        generated_files = tool_result.get('generated_files', [])

                    # 将工具的输出包装成模型可以理解的格式
                    tool_response_parts.append(types.Part.from_function_response(
                        name='segment_image',
                        response={'result': tool_result} # 将整个结果字典返回
                    ))

            # 4. 第二次调用：将工具结果发回给模型，让它生成最终的自然语言回复
            if tool_response_parts:
                logger.info("第二步：将工具执行结果返回给模型...")
                # 构建第二次调用的上下文
                second_call_contents = full_contents + [model_response_content, types.Content(role='tool', parts=tool_response_parts)]
                
                final_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=second_call_contents,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT) # 第二次调用不需要工具
                )
                logger.debug(f"第二次调用响应: {final_response}")
                final_text = final_response.text
                # 更新历史记录用的模型响应
                if final_response.candidates:
                    model_response_content = final_response.candidates[0].content
        else:
            # 如果模型没有请求工具调用，直接使用它的文本回复
            logger.info("模型未请求工具调用，直接使用文本回复。")
            final_text = response.text
            model_response_content = response.candidates[0].content if response.candidates else None

        # 如果在所有步骤后仍然没有文本，则提供一个最终的默认回复
        if not final_text:
             final_text = "我收到消息了，但好像没什么需要我做的。"

        # 更新历史记录
        user_content_for_history = types.Content(role='user', parts=[types.Part.from_text(text=user_message)])
        if model_response_content:
            conversation_sessions[contact_name].extend([user_content_for_history, model_response_content])

        logger.info(f"最终回复文本: '{final_text}', 待发送文件: {len(generated_files)}个")
        _save_sessions(conversation_sessions)
        return final_text, generated_files

    except Exception as e:
        logger.error(f"调用 Gemini API 或工具时出错: {e}", exc_info=True)
        return "抱歉，我现在有点忙，请稍后再试吧。", []

def clear_history(contact_name: str) -> bool:
    """
    清除指定联系人的对话历史记录。

    Args:
        contact_name (str): 要清除历史记录的联系人名称。

    Returns:
        bool: 如果成功找到并清除了历史记录，则返回 True，否则返回 False。
    """
    if contact_name in conversation_sessions:
        logger.info(f"正在为 '{contact_name}' 清除历史记录...")
        conversation_sessions[contact_name] = []
        _save_sessions(conversation_sessions)
        logger.info(f"'{contact_name}' 的历史记录已成功清除。")
        return True
    else:
        logger.warning(f"尝试清除一个不存在的历史记录: '{contact_name}'")
        return False
