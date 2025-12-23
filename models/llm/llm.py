import logging
from collections.abc import Generator
from typing import Optional, Union

from openai import OpenAI, APIError, AuthenticationError
from dify_plugin import LargeLanguageModel
from dify_plugin.entities import I18nObject
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    ModelType,
)
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
    LLMUsage,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageTool,
    SystemPromptMessage,
    UserPromptMessage,
    ToolPromptMessage,
)

logger = logging.getLogger(__name__)

ATLASCLOUD_API_BASE = "https://api.atlascloud.ai/v1"


class DifyPluginAtlascloudLargeLanguageModel(LargeLanguageModel):
    """
    Model class for AtlasCloud large language model.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model
        """
        client = self._get_client(credentials)
        messages = self._convert_messages(prompt_messages)

        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **model_parameters,
        }
        if stop:
            params["stop"] = stop
        if tools:
            params["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        if user:
            params["user"] = user

        try:
            if stream:
                return self._invoke_stream(client, params)
            else:
                return self._invoke_sync(client, params, model)
        except Exception as e:
            raise self._transform_invoke_error(e)

    def _get_client(self, credentials: dict) -> OpenAI:
        api_key = credentials.get("api_key") or credentials.get("openai_api_key")
        return OpenAI(api_key=api_key, base_url=ATLASCLOUD_API_BASE)

    def _convert_messages(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        messages = []
        for msg in prompt_messages:
            if isinstance(msg, SystemPromptMessage):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, UserPromptMessage):
                if isinstance(msg.content, str):
                    messages.append({"role": "user", "content": msg.content})
                else:
                    # Handle multimodal content
                    content = []
                    for item in msg.content:
                        if item.type == "text":
                            content.append({"type": "text", "text": item.data})
                        elif item.type == "image":
                            content.append({"type": "image_url", "image_url": {"url": item.data}})
                    messages.append({"role": "user", "content": content})
            elif isinstance(msg, AssistantPromptMessage):
                m = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    m["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                messages.append(m)
            elif isinstance(msg, ToolPromptMessage):
                messages.append({"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content})
        return messages

    def _invoke_stream(self, client: OpenAI, params: dict) -> Generator[LLMResultChunk, None, None]:
        response = client.chat.completions.create(**params)
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield LLMResultChunk(
                    model=chunk.model,
                    prompt_messages=[],
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=delta.content),
                    ),
                )
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield LLMResultChunk(
                        model=chunk.model,
                        prompt_messages=[],
                        delta=LLMResultChunkDelta(
                            index=tc.index,
                            message=AssistantPromptMessage(
                                content="",
                                tool_calls=[AssistantPromptMessage.ToolCall(
                                    id=tc.id or "",
                                    type="function",
                                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                        name=tc.function.name or "",
                                        arguments=tc.function.arguments or "",
                                    ),
                                )],
                            ),
                        ),
                    )
            if chunk.choices[0].finish_reason:
                yield LLMResultChunk(
                    model=chunk.model,
                    prompt_messages=[],
                    delta=LLMResultChunkDelta(
                        index=0,
                        message=AssistantPromptMessage(content=""),
                        finish_reason=chunk.choices[0].finish_reason,
                        usage=self._get_usage(chunk) if hasattr(chunk, "usage") and chunk.usage else None,
                    ),
                )

    def _invoke_sync(self, client: OpenAI, params: dict, model: str) -> LLMResult:
        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        message = AssistantPromptMessage(content=choice.message.content or "")
        if choice.message.tool_calls:
            message.tool_calls = [
                AssistantPromptMessage.ToolCall(
                    id=tc.id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in choice.message.tool_calls
            ]
        return LLMResult(
            model=model,
            prompt_messages=[],
            message=message,
            usage=self._get_usage(response),
        )

    def _get_usage(self, response) -> LLMUsage:
        if not response.usage:
            return LLMUsage.empty_usage()
        return LLMUsage(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )

    def _transform_invoke_error(self, e: Exception) -> InvokeError:
        if isinstance(e, AuthenticationError):
            return InvokeAuthorizationError(str(e))
        elif isinstance(e, APIError):
            if e.status_code == 429:
                return InvokeRateLimitError(str(e))
            elif e.status_code >= 500:
                return InvokeServerUnavailableError(str(e))
            else:
                return InvokeBadRequestError(str(e))
        elif isinstance(e, (ConnectionError, TimeoutError)):
            return InvokeConnectionError(str(e))
        return InvokeError(str(e))

    def _invoke_error_mapping(self, e: Exception) -> InvokeError:
        """
        Map invoke errors to Dify plugin errors
        """
        return self._transform_invoke_error(e)

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages
        """
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            num_tokens = 0
            for msg in prompt_messages:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                num_tokens += len(encoding.encode(content)) + 4  # message overhead
            if tools:
                for tool in tools:
                    num_tokens += len(encoding.encode(tool.name + (tool.description or "") + str(tool.parameters)))
            return num_tokens
        except Exception:
            return 0

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials
        """
        try:
            logger.info(f"Validating credentials for model: {model}")
            client = self._get_client(credentials)
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            logger.info("Credentials validation successful")
        except AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise CredentialsValidateFailedError(f"Invalid API key: {e}")
        except Exception as e:
            logger.exception(f"Credentials validation failed: {e}")
            raise CredentialsValidateFailedError(str(e))

    def get_customizable_model_schema(
        self, model: str, credentials: dict
    ) -> AIModelEntity:
        """
        Get customizable model schema
        """
        return AIModelEntity(
            model=model,
            label=I18nObject(zh_Hans=model, en_US=model),
            model_type=ModelType.LLM,
            features=[],
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={"mode": "chat", "context_size": 128000},
            parameter_rules=[],
        )
