import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from google.genai import types

# 1. 创建一个模拟的 client 实例
mock_generate_content_func = AsyncMock()
mock_client = MagicMock()
mock_client.aio.models.generate_content = mock_generate_content_func

# 2. 使用 patch 来确保当 `gemini_handler` 尝试创建 `genai.Client` 时，它得到的是我们伪造的 `mock_client`
@patch('gemini_handler.genai.Client', return_value=mock_client)
@patch('gemini_handler._load_sessions', return_value={})
def reload_handler_with_mocks(mock_load_sessions, mock_client_constructor):
    import importlib
    import gemini_handler
    importlib.reload(gemini_handler)
    return gemini_handler

gemini_handler = reload_handler_with_mocks()

@pytest.fixture(autouse=True)
def reset_mocks():
    yield
    mock_generate_content_func.reset_mock()
    mock_generate_content_func.side_effect = None

# --- 测试 _intent_router_async ---

@pytest.mark.asyncio
async def test_intent_router_function_call_intent():
    mock_response = MagicMock()
    mock_response.text = "FUNCTION_CALL_INTENT"
    mock_generate_content_func.return_value = mock_response
    intent = await gemini_handler._intent_router_async("抠图", [])
    assert intent == "FUNCTION_CALL_INTENT"

@pytest.mark.asyncio
async def test_intent_router_grounding_intent():
    mock_response = MagicMock()
    mock_response.text = "GROUNDING_INTENT"
    mock_generate_content_func.return_value = mock_response
    intent = await gemini_handler._intent_router_async("今天天气怎么样？", [])
    assert intent == "GROUNDING_INTENT"

@pytest.mark.asyncio
async def test_intent_router_general_conversation_intent():
    mock_response = MagicMock()
    mock_response.text = "GENERAL_CONVERSATION_INTENT"
    mock_generate_content_func.return_value = mock_response
    intent = await gemini_handler._intent_router_async("你好啊", [])
    assert intent == "GENERAL_CONVERSATION_INTENT"

@pytest.mark.asyncio
async def test_intent_router_fallback_on_unrecognized_intent():
    mock_response = MagicMock()
    mock_response.text = "SOME_WEIRD_INTENT"
    mock_generate_content_func.return_value = mock_response
    intent = await gemini_handler._intent_router_async("奇怪的请求", [])
    assert intent == "GENERAL_CONVERSATION_INTENT"

@pytest.mark.asyncio
async def test_intent_router_fallback_on_api_error():
    mock_generate_content_func.side_effect = Exception("API Error")
    intent = await gemini_handler._intent_router_async("任何请求", [])
    assert intent == "GENERAL_CONVERSATION_INTENT"

# --- 测试 _execute_function_call_flow_async ---

@pytest.mark.asyncio
@patch('gemini_handler.segment_image_async', new_callable=AsyncMock)
async def test_execute_function_call_flow_success(mock_segment_image):
    first_response = MagicMock()
    mock_fc = types.FunctionCall(name='segment_image_async', args={'chat_name': 'test_chat', 'user_prompt': '抠图'})
    part = types.Part(function_call=mock_fc)
    # 最终修复：移除 Pylance 报错的 finish_reason
    first_response.candidates = [types.Candidate(content=types.Content(parts=[part]))]
    first_response.function_calls = [mock_fc]

    second_response = MagicMock()
    second_response.text = "图片已经抠好啦！"
    second_response.candidates = [types.Candidate(content=types.Content(parts=[types.Part(text="图片已经抠好啦！")]))]

    mock_generate_content_func.side_effect = [first_response, second_response]

    mock_segment_image.return_value = {
        'status': 'success',
        'message': '成功生成了1张图片。',
        'generated_files': ['/path/to/image.png']
    }

    final_text, generated_files, _ = await gemini_handler._execute_function_call_flow_async(
        [types.Content(parts=[types.Part(text="用户消息")])], "系统提示", "test_chat", "抠图"
    )

    assert final_text == "图片已经抠好啦！"
    assert generated_files == ['/path/to/image.png']
    mock_segment_image.assert_called_once_with(chat_name='test_chat', user_prompt='抠图')

@pytest.mark.asyncio
async def test_execute_function_call_flow_no_call():
    response = MagicMock()
    response.text = "你好"
    response.function_calls = None
    response.candidates = [types.Candidate(content=types.Content(parts=[types.Part(text="你好")]))]
    mock_generate_content_func.return_value = response

    final_text, generated_files, _ = await gemini_handler._execute_function_call_flow_async(
        [types.Content(parts=[types.Part(text="用户消息")])], "系统提示", "test_chat", "你好"
    )

    assert final_text == "你好"
    assert generated_files == []

# --- 测试 _execute_grounding_flow_async ---
@pytest.mark.asyncio
async def test_execute_grounding_flow_success():
    mock_response = MagicMock()
    mock_response.text = "这是来自谷歌搜索的答案。"
    candidate = types.Candidate(content=types.Content(parts=[types.Part(text=mock_response.text)]), grounding_metadata=None)
    mock_response.candidates = [candidate]
    
    mock_generate_content_func.return_value = mock_response

    final_text, _ = await gemini_handler._execute_grounding_flow_async([types.Content(parts=[types.Part(text="用户查询")])], "系统提示")

    assert "这是来自谷歌搜索的答案。" in final_text

# --- 测试 _execute_general_conversation_flow_async ---
@pytest.mark.asyncio
async def test_execute_general_conversation_flow_success():
    mock_response = MagicMock()
    mock_response.text = "好的，没问题。"
    mock_response.candidates = [types.Candidate(content=types.Content(parts=[types.Part(text="好的，没问题。")]))]
    mock_generate_content_func.return_value = mock_response

    final_text, _ = await gemini_handler._execute_general_conversation_flow_async([types.Content(parts=[types.Part(text="你好")])], "系统提示")

    assert final_text == "好的，没问题。"