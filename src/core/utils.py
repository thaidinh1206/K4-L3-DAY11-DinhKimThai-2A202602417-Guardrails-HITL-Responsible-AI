"""
Lab 11 — Helper Utilities
"""
from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    plugins_list = list(getattr(runner, "plugins", []))
    pm = getattr(runner, "plugin_manager", None)
    if pm and getattr(pm, "plugins", None):
        for p in pm.plugins:
            if p not in plugins_list:
                plugins_list.append(p)

    # Check input plugins first to short-circuit if blocked by guardrails
    for plugin in plugins_list:
        cb = getattr(plugin, "on_user_message_callback", None)
        if cb is not None:
            user_content = types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )
            class _DummyCtx:
                pass
            try:
                blocked = await cb(invocation_context=_DummyCtx(), user_message=user_content)
                if blocked is not None and getattr(blocked, "parts", None):
                    text = "".join(p.text for p in blocked.parts if getattr(p, "text", None))
                    return text, None
            except Exception:
                pass

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    for plugin in plugins_list:
        out_cb = getattr(plugin, "after_model_callback", None)
        if out_cb is not None:
            class _DummyResp:
                def __init__(self, t):
                    self.content = types.Content(
                        role="model", parts=[types.Part.from_text(text=t)]
                    )

            try:
                resp_obj = _DummyResp(final_response)
                res = await out_cb(callback_context=None, llm_response=resp_obj)
                if res and hasattr(res, "content") and res.content and res.content.parts:
                    final_response = "".join(
                        p.text for p in res.content.parts if getattr(p, "text", None)
                    )
            except Exception:
                pass

    return final_response, session
