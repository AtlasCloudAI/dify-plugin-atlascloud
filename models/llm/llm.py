import codecs
import json
import logging
from collections.abc import Generator
from decimal import Decimal
from typing import Optional, Union, cast

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
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContent,
    PromptMessageContentType,
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
        messages = self._convert_messages(prompt_messages, credentials)

        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **model_parameters,
        }
        if stop:
            params["stop"] = stop

        # Handle tool calling based on function_calling_type
        function_calling_type = credentials.get("function_calling_type", "tool_call")
        if tools:
            if function_calling_type == "function_call":
                # Old OpenAI API style with functions
                params["functions"] = [
                    {"name": t.name, "description": t.description, "parameters": t.parameters}
                    for t in tools
                ]
            else:
                # New OpenAI API style with tools (default)
                params["tool_choice"] = "auto"
                params["tools"] = [
                    {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                    for t in tools
                ]
        if user:
            params["user"] = user

        try:
            if stream:
                return self._invoke_stream(client, params, prompt_messages, model, credentials)
            else:
                return self._invoke_sync(client, params, prompt_messages, model, credentials)
        except Exception as e:
            raise self._transform_invoke_error(e)

    def _get_client(self, credentials: dict) -> OpenAI:
        """
        Create OpenAI client with custom configuration
        """
        api_key = credentials.get("api_key") or credentials.get("openai_api_key")

        # Support extra headers if provided
        extra_headers = credentials.get("extra_headers", {})
        default_headers = {
            "Accept-Charset": "utf-8",
            **extra_headers
        }

        # Support custom timeout
        timeout = credentials.get("timeout", 300)

        return OpenAI(
            api_key=api_key,
            base_url=ATLASCLOUD_API_BASE,
            default_headers=default_headers,
            timeout=timeout
        )

    def _convert_messages(self, prompt_messages: list[PromptMessage], credentials: Optional[dict] = None) -> list[dict]:
        """
        Convert prompt messages to OpenAI API format
        """
        messages = []
        function_calling_type = credentials.get("function_calling_type", "tool_call") if credentials else "tool_call"

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
                        if item.type == PromptMessageContentType.TEXT:
                            item = cast(PromptMessageContent, item)
                            content.append({"type": "text", "text": item.data})
                        elif item.type == PromptMessageContentType.IMAGE:
                            item = cast(ImagePromptMessageContent, item)
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": item.data, "detail": item.detail.value}
                            })
                    messages.append({"role": "user", "content": content})
            elif isinstance(msg, AssistantPromptMessage):
                m = {"role": "assistant", "content": msg.content or ""}
                if msg.tool_calls:
                    if function_calling_type == "tool_call":
                        # New style: tool_calls
                        m["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                            }
                            for tc in msg.tool_calls
                        ]
                    elif function_calling_type == "function_call":
                        # Old style: function_call (only first tool call)
                        first_tool_call = msg.tool_calls[0]
                        m["function_call"] = {
                            "name": first_tool_call.function.name,
                            "arguments": first_tool_call.function.arguments
                        }
                messages.append(m)
            elif isinstance(msg, ToolPromptMessage):
                if function_calling_type == "tool_call":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content
                    })
                elif function_calling_type == "function_call":
                    messages.append({
                        "role": "function",
                        "name": msg.tool_call_id,
                        "content": msg.content
                    })
        return messages

    def _invoke_stream(
        self,
        client: OpenAI,
        params: dict,
        prompt_messages: list[PromptMessage],
        model: str,
        credentials: dict
    ) -> Generator[LLMResultChunk, None, None]:
        """
        Handle streaming response with robust parsing
        """
        response = client.chat.completions.create(**params)

        full_assistant_content = ""
        chunk_index = 0
        tools_calls: list[AssistantPromptMessage.ToolCall] = []
        function_calling_type = credentials.get("function_calling_type", "tool_call")
        finish_reason = None
        last_chunk = None

        def increase_tool_call(new_tool_calls: list[AssistantPromptMessage.ToolCall]):
            """Incrementally accumulate tool calls"""
            def get_tool_call(tool_call_id: str):
                if not tool_call_id:
                    return tools_calls[-1] if tools_calls else None

                tool_call = next((tc for tc in tools_calls if tc.id == tool_call_id), None)
                if tool_call is None:
                    tool_call = AssistantPromptMessage.ToolCall(
                        id=tool_call_id,
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="", arguments=""),
                    )
                    tools_calls.append(tool_call)
                return tool_call

            for new_tool_call in new_tool_calls:
                # Get or create tool call
                tool_call = get_tool_call(new_tool_call.id if hasattr(new_tool_call, 'id') else "")
                if tool_call is None:
                    continue

                # Update tool call incrementally
                if hasattr(new_tool_call, 'id') and new_tool_call.id:
                    tool_call.id = new_tool_call.id
                if hasattr(new_tool_call, 'type') and new_tool_call.type:
                    tool_call.type = new_tool_call.type
                if new_tool_call.function.name:
                    tool_call.function.name = new_tool_call.function.name
                if new_tool_call.function.arguments:
                    tool_call.function.arguments += new_tool_call.function.arguments

        for chunk in response:
            last_chunk = chunk
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Handle content delta
            if delta.content:
                full_assistant_content += delta.content
                yield LLMResultChunk(
                    model=chunk.model,
                    prompt_messages=prompt_messages,
                    delta=LLMResultChunkDelta(
                        index=chunk_index,
                        message=AssistantPromptMessage(content=delta.content),
                    ),
                )
                chunk_index += 1

            # Handle tool calls delta
            assistant_message_tool_calls = None
            if hasattr(delta, 'tool_calls') and delta.tool_calls and function_calling_type == "tool_call":
                assistant_message_tool_calls = delta.tool_calls
            elif hasattr(delta, 'function_call') and delta.function_call and function_calling_type == "function_call":
                # Convert function_call to tool_calls format
                assistant_message_tool_calls = [
                    type('ToolCall', (), {
                        'id': 'call_function',
                        'type': 'function',
                        'function': delta.function_call
                    })()
                ]

            if assistant_message_tool_calls:
                # Convert to our format
                tool_calls_to_add = []
                for tc in assistant_message_tool_calls:
                    tool_calls_to_add.append(
                        AssistantPromptMessage.ToolCall(
                            id=getattr(tc, 'id', '') or '',
                            type=getattr(tc, 'type', 'function'),
                            function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                name=getattr(tc.function, 'name', '') or '',
                                arguments=getattr(tc.function, 'arguments', '') or '',
                            ),
                        )
                    )
                increase_tool_call(tool_calls_to_add)

        # Send final chunk with usage and finish_reason
        # Calculate usage using base class method
        usage = None
        if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
            # Use API-provided usage if available
            prompt_tokens = last_chunk.usage.prompt_tokens or 0
            completion_tokens = last_chunk.usage.completion_tokens or 0
            usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)
        else:
            # Fallback: calculate tokens manually
            prompt_tokens = self._num_tokens_from_messages(model, prompt_messages, None, credentials)
            completion_tokens = self._num_tokens_from_string(model, full_assistant_content)
            usage = self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

        # Build final message with tool calls if any
        final_message = AssistantPromptMessage(content="")
        if tools_calls:
            final_message.tool_calls = tools_calls

        yield LLMResultChunk(
            model=last_chunk.model if last_chunk else model,
            prompt_messages=prompt_messages,
            delta=LLMResultChunkDelta(
                index=chunk_index,
                message=final_message,
                finish_reason=finish_reason or "stop",
                usage=usage,
            ),
        )

    def _invoke_sync(
        self,
        client: OpenAI,
        params: dict,
        prompt_messages: list[PromptMessage],
        model: str,
        credentials: dict
    ) -> LLMResult:
        """
        Handle synchronous response
        """
        response = client.chat.completions.create(**params)
        choice = response.choices[0]
        message = AssistantPromptMessage(content=choice.message.content or "")

        function_calling_type = credentials.get("function_calling_type", "tool_call")

        # Handle tool calls
        if function_calling_type == "tool_call" and choice.message.tool_calls:
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
        elif function_calling_type == "function_call" and hasattr(choice.message, 'function_call') and choice.message.function_call:
            # Convert function_call to tool_calls format
            message.tool_calls = [
                AssistantPromptMessage.ToolCall(
                    id="call_function",
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=choice.message.function_call.name,
                        arguments=choice.message.function_call.arguments,
                    ),
                )
            ]

        return LLMResult(
            model=response.model,
            prompt_messages=prompt_messages,
            message=message,
            usage=self._get_usage(response, model, credentials),
        )

    def _get_usage(self, response, model: str, credentials: dict) -> LLMUsage:
        """
        Extract usage information from response and calculate pricing
        """
        if not response.usage:
            return LLMUsage.empty_usage()

        # Get token counts from API response
        prompt_tokens = response.usage.prompt_tokens or 0
        completion_tokens = response.usage.completion_tokens or 0

        # Use base class method to calculate pricing based on model configuration
        return self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

    def _transform_invoke_error(self, e: Exception) -> InvokeError:
        """
        Transform various errors to Dify invoke errors
        """
        if isinstance(e, AuthenticationError):
            return InvokeAuthorizationError(str(e))
        elif isinstance(e, APIError):
            if hasattr(e, 'status_code'):
                if e.status_code == 400:
                    return InvokeBadRequestError(str(e))
                elif e.status_code == 401:
                    return InvokeAuthorizationError(str(e))
                elif e.status_code == 429:
                    return InvokeRateLimitError(str(e))
                elif e.status_code >= 500:
                    return InvokeServerUnavailableError(str(e))
                else:
                    return InvokeBadRequestError(str(e))
            else:
                return InvokeServerUnavailableError(str(e))
        elif isinstance(e, (ConnectionError, TimeoutError)):
            return InvokeConnectionError(str(e))
        elif "timeout" in str(e).lower():
            return InvokeConnectionError(f"Request timeout: {str(e)}")
        elif "connection" in str(e).lower():
            return InvokeConnectionError(f"Connection error: {str(e)}")
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
        return self._num_tokens_from_messages(model, prompt_messages, tools, credentials)

    def _get_num_tokens_by_gpt2(self, text: str) -> int:
        """
        Use GPT2 tokenizer to calculate num tokens (fallback method)
        """
        try:
            from transformers import GPT2TokenizerFast
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception:
            # If transformers is not available, use rough estimation
            # Approximately 1 token per 4 characters for English text
            return max(1, len(text) // 4)

    def _num_tokens_from_string(
        self,
        model: str,
        text: Union[str, list[PromptMessageContent]],
        tools: Optional[list[PromptMessageTool]] = None
    ) -> int:
        """
        Calculate num tokens for text content
        """
        if isinstance(text, str):
            full_text = text
        else:
            full_text = ""
            for message_content in text:
                if message_content.type == PromptMessageContentType.TEXT:
                    message_content = cast(PromptMessageContent, message_content)
                    full_text += message_content.data

        num_tokens = self._get_num_tokens_by_gpt2(full_text)

        if tools:
            num_tokens += self._num_tokens_for_tools(tools)

        return num_tokens

    def _num_tokens_from_messages(
        self,
        model: str,
        messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
        credentials: Optional[dict] = None,
    ) -> int:
        """
        Calculate num tokens for messages using GPT2 tokenizer
        """
        # Try tiktoken first for more accurate counting
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")

            tokens_per_message = 3
            tokens_per_name = 1
            num_tokens = 0

            messages_dict = [self._convert_messages([m], credentials)[0] for m in messages]
            for message in messages_dict:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    if isinstance(value, list):
                        text = ""
                        for item in value:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text += item.get("text", "")
                        value = text

                    if key == "tool_calls":
                        for tool_call in value:
                            for t_key, t_value in tool_call.items():
                                num_tokens += len(encoding.encode(str(t_key)))
                                if t_key == "function":
                                    for f_key, f_value in t_value.items():
                                        num_tokens += len(encoding.encode(str(f_key)))
                                        num_tokens += len(encoding.encode(str(f_value)))
                                else:
                                    num_tokens += len(encoding.encode(str(t_value)))
                    else:
                        num_tokens += len(encoding.encode(str(value)))

                    if key == "name":
                        num_tokens += tokens_per_name

            num_tokens += 3  # every reply is primed with assistant

            if tools:
                num_tokens += self._num_tokens_for_tools(tools)

            return num_tokens
        except Exception:
            # Fallback to GPT2 tokenizer
            tokens_per_message = 3
            num_tokens = 0

            for msg in messages:
                num_tokens += tokens_per_message
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                num_tokens += self._get_num_tokens_by_gpt2(content)

            if tools:
                num_tokens += self._num_tokens_for_tools(tools)

            return num_tokens

    def _num_tokens_for_tools(self, tools: list[PromptMessageTool]) -> int:
        """
        Calculate num tokens for tool calling
        """
        num_tokens = 0
        for tool in tools:
            # Count tokens for each tool
            num_tokens += self._get_num_tokens_by_gpt2("type")
            num_tokens += self._get_num_tokens_by_gpt2("function")
            num_tokens += self._get_num_tokens_by_gpt2("name")
            num_tokens += self._get_num_tokens_by_gpt2(tool.name)
            num_tokens += self._get_num_tokens_by_gpt2("description")
            num_tokens += self._get_num_tokens_by_gpt2(tool.description or "")

            # Count tokens for parameters
            parameters = tool.parameters
            num_tokens += self._get_num_tokens_by_gpt2("parameters")
            if "title" in parameters:
                num_tokens += self._get_num_tokens_by_gpt2("title")
                num_tokens += self._get_num_tokens_by_gpt2(str(parameters.get("title", "")))
            if "type" in parameters:
                num_tokens += self._get_num_tokens_by_gpt2("type")
                num_tokens += self._get_num_tokens_by_gpt2(str(parameters.get("type", "")))
            if "properties" in parameters:
                num_tokens += self._get_num_tokens_by_gpt2("properties")
                for key, value in parameters.get("properties", {}).items():
                    num_tokens += self._get_num_tokens_by_gpt2(key)
                    for field_key, field_value in value.items():
                        num_tokens += self._get_num_tokens_by_gpt2(field_key)
                        if field_key == "enum":
                            for enum_field in field_value:
                                num_tokens += 3
                                num_tokens += self._get_num_tokens_by_gpt2(str(enum_field))
                        else:
                            num_tokens += self._get_num_tokens_by_gpt2(str(field_value))
            if "required" in parameters:
                num_tokens += self._get_num_tokens_by_gpt2("required")
                for required_field in parameters.get("required", []):
                    num_tokens += 3
                    num_tokens += self._get_num_tokens_by_gpt2(str(required_field))

        return num_tokens

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
