"""Secret-free provider diagnostics and traceback logging.

Never serialize upstream exception messages, response bodies, locals, or headers.
Provider responses can echo credentials (including partially masked keys).
"""
import logging
import traceback


def provider_failure(error: Exception) -> tuple[str, str, bool]:
    status = getattr(error, "status_code", None)
    code = str(getattr(error, "code", "") or "").lower()
    message = str(error).lower()  # Inspect only; never return/log this value.
    if status == 401 or code in {"invalid_api_key", "api_key_invalid"} or any(
        word in message for word in ("authentication", "unauthorized", "api key not valid", "incorrect api key", "api_key_invalid")
    ):
        return "provider_authentication", "rejected its configured API key. Replace the credential in Settings.", False
    if status == 403 or code == "permission_denied":
        return "provider_permission", "denied access. Check the account and model permissions.", False
    if code == "insufficient_quota" or "quota" in message:
        return "provider_quota", "account quota was exceeded. Check billing and quota.", False
    if status == 429 or "rate limit" in message:
        return "provider_rate_limit", "rate limit was exceeded. Retry later or reduce request volume.", True
    if "tool" in message and ("not support" in message or "unsupported" in message):
        return "invalid_model", "model does not support tool calling. Choose a tool-capable model in Settings.", False
    if status == 404 or code == "model_not_found" or any(word in message for word in ("unsupported", "not support", "model_not_found", "invalid model", "model does not exist")):
        return "invalid_model", "could not use the configured model or operation. Check the model ID, capabilities, and permissions in Settings.", False
    if status in {408, 409} or (isinstance(status, int) and status >= 500) or any(
        word in type(error).__name__.lower() for word in ("timeout", "connection")
    ):
        return "provider_unavailable", "is temporarily unavailable. Retry later.", True
    return "provider_request", "rejected the request. Check the model configuration and server logs.", False


def log_safe_exception(logger: logging.Logger, error: Exception, context: str) -> None:
    """Retain every traceback frame and exception type without secret-bearing text."""
    chain = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames = traceback.extract_tb(current.__traceback__)
        chain.append(type(current).__name__ + "\n" + "\n".join(
            f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}' for frame in frames
        ))
        current = current.__cause__ or current.__context__
    logger.error("%s\n%s", context, "\nCaused by:\n".join(chain))
