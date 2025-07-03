import json
import os
import PIL.Image
import io
import base64
import binascii
import numpy as np
from typing import Optional, Any, List, Dict
from PIL import Image, ImageDraw
import time
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_BASE_URL, SYSTEM_PROMPT, HISTORY_DIR, MAX_HISTORY_TURNS, IMAGE_CONTEXT_TTL, ENABLE_GOOGLE_SEARCH
from logger import logger

# --- 全局设置 ---

# 确保历史记录目录存在
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

# 配置 API 客户端
if GEMINI_BASE_URL:
    cleaned_url = GEMINI_BASE_URL.strip().rstrip('/')
    logger.debug(f"正在使用自定义端点: {cleaned_url}")
    http_opts = types.HttpOptions(base_url = cleaned_url)
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_opts)
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

# --- 图片上下文管理 ---
last_image_context = {}

def update_image_context(chat_name: str, path: str, timestamp: float):
    global last_image_context
    last_image_context[chat_name] = {'path': path, 'timestamp': timestamp}
    logger.info(f"已为 [{chat_name}] 更新图片上下文。")

def get_image_path_from_context(chat_name: str) -> Optional[str]:
    context = last_image_context.get(chat_name)
    if context and time.time() - context.get('timestamp', 0) < IMAGE_CONTEXT_TTL:
        return context.get('path')
    elif context:
        del last_image_context[chat_name]
        logger.info(f"'{chat_name}' 的图片上下文已过期并被清除。")
    return None

# --- 对话历史记录 (JSON 实现) ---
SESSIONS_FILE = os.path.join(HISTORY_DIR, 'sessions.json')

def _content_to_dict(content: types.Content) -> Dict[str, Any]:
    parts_list = []
    if content.parts:
        for part in content.parts:
            if hasattr(part, 'text'):
                parts_list.append({'text': part.text})
    return {'role': content.role, 'parts': parts_list}

def _dict_to_content(data: Dict[str, Any]) -> types.Content:
    parts_list = []
    for part_dict in data.get('parts', []):
        if 'text' in part_dict:
            parts_list.append(types.Part.from_text(text=part_dict['text']))
    return types.Content(role=data.get('role'), parts=parts_list)

def _load_sessions() -> Dict[str, Any]:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, 'r', encoding='utf-8') as f:
                logger.debug(f"从 '{SESSIONS_FILE}' 加载历史记录...")
                sessions_dict = json.load(f)
                return {
                    chat_name: [_dict_to_content(msg) for msg in messages]
                    for chat_name, messages in sessions_dict.items()
                }
        except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
            logger.error(f"加载或解析历史记录文件失败: {e}。将创建新的历史记录文件。")
    return {}

def _save_sessions(sessions: Dict[str, Any]):
    try:
        with open(SESSIONS_FILE, 'w', encoding='utf-8') as f:
            sessions_dict = {
                chat_name: [_content_to_dict(msg) for msg in messages]
                for chat_name, messages in sessions.items()
            }
            json.dump(sessions_dict, f, ensure_ascii=False, indent=4)
            logger.debug(f"历史记录已成功保存到 '{SESSIONS_FILE}'。")
    except Exception as e:
        logger.error(f"保存历史记录失败: {e}")

conversation_sessions = _load_sessions()

# --- 工具定义 ---

def _parse_json_from_gemini(json_output: str) -> str:
    if "```json" in json_output:
        json_output = json_output.split("```json")[1].split("```")[0]
    return json_output.strip()

async def segment_image_async(chat_name: str, user_prompt: str) -> dict:
    output_dir = "segmentation_outputs"
    os.makedirs(output_dir, exist_ok=True)
    context = last_image_context.get(chat_name)
    if not context or time.time() - context.get('timestamp', 0) > IMAGE_CONTEXT_TTL:
        return {'status': 'failure', 'message': '我需要你先发一张图片，然后我才能处理。'}
    image_path = context['path']
    try:
        img = PIL.Image.open(image_path)
        img.thumbnail((1024, 1024), PIL.Image.Resampling.LANCZOS)
        prompt = f'根据用户的指令 "{user_prompt}"，对图像中的对象进行分割。输出一个 JSON 列表，每个条目包含 "box_2d", "mask", 和 "label"。'
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0), # 为分割任务禁用思考，以提升效果
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
        response = await client.aio.models.generate_content(model='gemini-2.5-flash', contents=[prompt, img], config=config)
        if not response.text:
            return {'status': 'failure', 'message': '抱歉，模型没有返回有效的 JSON 数据。'}
        logger.debug(f"Raw model response for JSON parsing: {response.text}")
        items = json.loads(_parse_json_from_gemini(response.text))
        if not items:
            return {'status': 'failure', 'message': '抱歉，我没能在图片中识别出任何可分割的对象。'}
        generated_files = []
        base_img_rgba = img.convert('RGBA')
        for i, item in enumerate(items):
            box, png_b64, label = item.get("box_2d"), item.get("mask"), item.get("label", "unknown")
            if not all([box, png_b64, label]): continue

            # Bug-fix: 如果模型返回一个列表，安全地取出第一个元素
            if isinstance(png_b64, list):
                if not png_b64: continue # 如果列表为空则跳过
                png_b64 = png_b64[0]

            try:
                logger.debug(f"Raw base64 content from model: {png_b64}")
                mask_data = base64.b64decode(png_b64.removeprefix("data:image/png;base64,"))
                mask_img = Image.open(io.BytesIO(mask_data))
            except (binascii.Error, ValueError): continue
            y0, x0, y1, x1 = [int(c / 1000 * s) for c, s in zip(box, [img.size[1], img.size[0], img.size[1], img.size[0]])]
            if y0 >= y1 or x0 >= x1: continue
            cutout_image = Image.new('RGBA', base_img_rgba.size, (0, 0, 0, 0))
            mask_resized = mask_img.resize((x1 - x0, y1 - y0), PIL.Image.Resampling.BILINEAR)
            full_mask = Image.new('L', base_img_rgba.size, 0)
            full_mask.paste(mask_resized, (x0, y0))
            cutout_image = Image.composite(base_img_rgba, cutout_image, full_mask)
            safe_label = "".join(c for c in label if c.isalnum())
            output_filename = f"{safe_label}_{i}.png"
            output_path = os.path.abspath(os.path.join(output_dir, output_filename))
            cutout_image.save(output_path)
            generated_files.append(output_path)
        if not generated_files:
            return {'status': 'failure', 'message': '我尝试处理了，但未能成功生成任何分割图片。'}
        return {'status': 'success', 'message': f"成功处理并生成了 {len(generated_files)} 张图片。", 'generated_files': generated_files}
    except Exception as e:
        logger.error(f"工具 'segment_image' 执行失败: {e}", exc_info=True)
        return {'status': 'error', 'message': "抱歉，处理图片时遇到了一个内部错误，请稍后再试。"}

segment_image_declaration = types.FunctionDeclaration(
    name='segment_image_async',
    description="当用户想要从图片中提取、分割或抠出某些物体时，调用此工具。工具会自动查找当前聊天会话中的最新图片进行处理。",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            'chat_name': types.Schema(type=types.Type.STRING, description="当前聊天的名称，用于查找图片上下文。"),
            'user_prompt': types.Schema(type=types.Type.STRING, description="用户的具体指令，描述了想要分割出的对象。例如：“把图里的猫抠出来”。")
        },
        required=['chat_name', 'user_prompt']
    )
)
segment_image_tool = types.Tool(function_declarations=[segment_image_declaration])

# --- v2.0 智能意图路由器 ---

async def _intent_router_async(user_query: str, history: List[types.Content]) -> str:
    """使用轻量级LLM对用户意图进行分类。"""
    try:
        history_str = json.dumps([_content_to_dict(msg) for msg in history], ensure_ascii=False, indent=2)
        router_prompt = f"""
分析以下用户查询和对话历史，判断其主要意图。
从以下四种意图中选择一个，并只返回意图的名称：
1. FUNCTION_CALL_INTENT: 用户明确要求执行一个动作（如“把图里的猫抠出来”、“分割图片”）。
2. GROUNDING_INTENT: 用户明确要求基于提供的文档或知识库进行回答（如“根据XX文档，总结一下...”）。当配置中启用Google搜索时，这也包括需要联网查询的请求。
3. HYBRID_INTENT: 用户的请求是多步骤的，既需要从知识库获取信息，又需要基于该信息执行动作。
4. GENERAL_CONVERSATION_INTENT: 普通聊天、问候或不属于以上任何一种的查询。

请仅分析 <user_query> 标签内的内容。
<user_query>
{user_query}
</user_query>
对话历史:
```json
{history_str}
```
意图是:"""
        
        response = await client.aio.models.generate_content(
            model='gemini-2.5-flash',
            contents=[router_prompt],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        
        intent = response.text.strip() if response.text else ""
        logger.info(f"意图路由器识别结果: '{intent}'")
        
        valid_intents = ["FUNCTION_CALL_INTENT", "GROUNDING_INTENT", "HYBRID_INTENT", "GENERAL_CONVERSATION_INTENT"]
        if intent in valid_intents:
            return intent
        else:
            logger.warning(f"无法识别的意图 '{intent}'，将回退到通用对话。")
            return "GENERAL_CONVERSATION_INTENT"
            
    except Exception as e:
        logger.error(f"意图路由器执行失败: {e}", exc_info=True)
        return "GENERAL_CONVERSATION_INTENT"

def _format_citations_for_wechat(response) -> str:
    """从模型响应中提取并格式化引用信息。"""
    final_text = response.text
    try:
        if hasattr(response, 'candidates') and response.candidates:
            metadata = getattr(response.candidates[0], 'grounding_metadata', None)
            if metadata and hasattr(metadata, 'grounding_chunks') and metadata.grounding_chunks:
                citation_texts = [f"[{i+1}] {getattr(chunk.web, 'title', '未知标题')}: {getattr(chunk.web, 'uri', 'N/A')}" for i, chunk in enumerate(metadata.grounding_chunks) if hasattr(chunk, 'web')]
                if citation_texts:
                    final_text += "\n\n---\n信息来源:\n" + "\n".join(citation_texts)
    except Exception as e:
        logger.error(f"格式化引用信息时出错: {e}")
    return final_text

async def _execute_grounding_flow_async(full_contents: List[Any], system_prompt: str) -> tuple[str, Any]:
    """执行接地流程。"""
    logger.info("执行接地流程...")
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[types.Tool(google_search=types.GoogleSearch())],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.NONE
            )
        )
    )
    response = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_contents,
        config=config
    )
    final_text = _format_citations_for_wechat(response)
    model_response_content = response.candidates[0].content if response.candidates else None
    return final_text or "", model_response_content

async def _execute_function_call_flow_async(full_contents: List[Any], system_prompt: str, contact_name: str, user_message: str) -> tuple[str, list, Optional[types.Content]]:
    """执行函数调用流程，并提供详细的日志记录。"""
    logger.info("--- 开始函数调用流程 ---")
    logger.debug(f"输入参数: contact_name='{contact_name}', user_message='{user_message[:50]}...'")
    
    available_tools = [segment_image_tool]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=available_tools,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
        )
    )
    
    logger.debug("步骤 1: 向模型发送初次请求，以确定是否需要调用工具。")
    response = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_contents,
        config=config
    )
    
    generated_files: List[str] = []
    model_response_content = response.candidates[0].content if response.candidates else None
    logger.debug(f"步骤 1.1: 收到初次响应。是否有函数调用请求? {'是' if response.function_calls else '否'}")

    if response.function_calls:
        logger.info("步骤 2: 模型请求调用工具。")
        logger.debug(f"请求调用的函数: {[fc.name for fc in response.function_calls]}")
        
        tool_response_parts = []
        for func_call in response.function_calls:
            if func_call.name == 'segment_image_async' and func_call.args:
                user_prompt_from_model = func_call.args.get('user_prompt', user_message)
                logger.debug(f"执行工具 'segment_image_async'，参数: chat_name='{contact_name}', user_prompt='{user_prompt_from_model}'")
                tool_result = await segment_image_async(chat_name=contact_name, user_prompt=user_prompt_from_model)
                logger.debug(f"工具 'segment_image_async' 返回结果: {tool_result}")
                
                if isinstance(tool_result, dict) and tool_result.get('status') == 'success':
                    generated_files.extend(tool_result.get('generated_files', []))
                    logger.info(f"工具执行成功，生成了 {len(tool_result.get('generated_files', []))} 个文件。")
                else:
                    logger.warning("工具执行失败或未生成文件。")
                    
                tool_response_parts.append(types.Part.from_function_response(name='segment_image_async', response=tool_result))
        
        if tool_response_parts:
            logger.info("步骤 3: 将工具执行结果返回给模型，以生成最终回复。")
            second_call_contents = full_contents + [model_response_content, types.Content(role='tool', parts=tool_response_parts)]
            
            final_response = await client.aio.models.generate_content(
                model='gemini-2.5-flash',
                contents=second_call_contents,
                config=types.GenerateContentConfig(system_instruction=system_prompt)
            )
            
            final_text = final_response.text
            model_response_content = final_response.candidates[0].content if final_response.candidates else None
            logger.info("步骤 3.1: 收到模型的最终回复。")
            logger.debug(f"最终回复文本: '{(final_text or '')[:100]}...'")
            logger.info("--- 函数调用流程成功结束 ---")
            return final_text or "", generated_files, model_response_content
        else:
            logger.warning("工具响应部分为空，流程中断。将回退到返回初次响应的文本。")

    logger.info("未检测到函数调用或工具执行失败。将返回模型的初次响应。")
    final_text = response.text
    logger.debug(f"返回的文本: '{(final_text or '')[:100]}...'")
    logger.info("--- 函数调用流程结束 (无实际调用) ---")
    return final_text or "", generated_files, model_response_content

async def _execute_general_conversation_flow_async(full_contents: List[Any], system_prompt: str) -> tuple[str, Optional[types.Content]]:
    """执行通用对话流程。"""
    logger.info("执行通用对话流程...")
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.NONE
            )
        )
    )
    response = await client.aio.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_contents,
        config=config
    )
    final_text = response.text
    model_response_content = response.candidates[0].content if response.candidates else None
    return final_text or "", model_response_content

# --- 主逻辑 ---

async def get_ai_response_async(contact_name: str, user_message: str, image_path: Optional[str] = None, is_group: bool = False, sender_name: Optional[str] = None) -> tuple[str, list]:
    try:
        # 步骤 1: 初始化和历史记录管理
        history = conversation_sessions.setdefault(contact_name, [])
        if MAX_HISTORY_TURNS > 0 and len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]
            conversation_sessions[contact_name] = history
        
        user_query_safe = f"<user_query>{user_message}</user_query>"
        final_user_message = f"请注意：你正在一个群聊中，当前向你提问的用户是“{sender_name}”。请结合上下文，并以对“{sender_name}”说话的口吻进行回复。\n\n用户的原始问题在下面的标签中：\n{user_query_safe}" if is_group and sender_name else user_query_safe
        
        prompt_parts: List[Any] = []
        if image_path:
            try:
                prompt_parts.append(PIL.Image.open(image_path))
            except Exception as e:
                return "抱歉，我无法处理您发送的图片，它可能已损坏或格式不支持。", []
        prompt_parts.append(final_user_message if final_user_message else " ")
        
        full_contents = history + prompt_parts
        
        # 步骤 2: 意图路由
        intent = await _intent_router_async(user_message, history)
        
        final_text = ""
        generated_files = []
        model_response_content = None

        # 步骤 3: 根据意图选择执行路径
        if intent == "FUNCTION_CALL_INTENT":
            final_text, generated_files, model_response_content = await _execute_function_call_flow_async(full_contents, SYSTEM_PROMPT, contact_name, user_message)
        
        elif intent == "GROUNDING_INTENT" and ENABLE_GOOGLE_SEARCH:
            final_text, model_response_content = await _execute_grounding_flow_async(full_contents, SYSTEM_PROMPT)

        elif intent == "HYBRID_INTENT" and ENABLE_GOOGLE_SEARCH:
            logger.info("执行混合流程...")
            # 1. 接地获取上下文
            grounding_text, _ = await _execute_grounding_flow_async(full_contents, SYSTEM_PROMPT)
            # 2. 增强查询并执行函数调用
            enhanced_query = f"基于以下背景信息：\n{grounding_text}\n\n请处理我的请求：\n<user_query>{user_message}</user_query>"
            enhanced_parts = [part for part in prompt_parts if not isinstance(part, str)] + [enhanced_query]
            enhanced_full_contents = history + enhanced_parts
            final_text, generated_files, model_response_content = await _execute_function_call_flow_async(enhanced_full_contents, SYSTEM_PROMPT, contact_name, user_message)

        else: # GENERAL_CONVERSATION_INTENT 或回退情况
            if intent != "GENERAL_CONVERSATION_INTENT":
                 logger.warning(f"意图 '{intent}' 的处理条件不满足（例如搜索被禁用），回退到通用对话。")
            final_text, model_response_content = await _execute_general_conversation_flow_async(full_contents, SYSTEM_PROMPT)

        # 步骤 4: 后处理和保存
        if not final_text:
             final_text = "我收到消息了，但好像没什么需要我做的。"

        if model_response_content:
            # 创建一个仅包含文本的用户消息部分用于存储
            user_part_for_history = types.Content(role='user', parts=[types.Part.from_text(text=user_message)])
            conversation_sessions[contact_name].extend([user_part_for_history, model_response_content])
        
        _save_sessions(conversation_sessions)
        return final_text, generated_files

    except Exception as e:
        logger.error(f"调用 Gemini API 或工具时出错: {e}", exc_info=True)
        return "抱歉，我现在有点忙，请稍后再试吧。", []

def clear_history(contact_name: str) -> bool:
    if contact_name in conversation_sessions:
        conversation_sessions[contact_name] = []
        _save_sessions(conversation_sessions)
        logger.info(f"'{contact_name}' 的历史记录已成功清除。")
        return True
    else:
        logger.warning(f"尝试清除一个不存在的历史记录: '{contact_name}'")
        return False
