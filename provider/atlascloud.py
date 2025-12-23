import logging
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class DifyPluginAtlascloudModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials
        if validate failed, raise exception

        :param credentials: provider credentials, credentials form defined in `provider_credential_schema`.
        """
        try:
            logger.info(f"Validating provider credentials: {list(credentials.keys())}")
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials(model='openai/gpt-4o-mini', credentials=credentials)
            logger.info("Provider credentials validation successful")
        except CredentialsValidateFailedError as ex:
            logger.error(f"Credentials validation failed: {ex}")
            raise ex
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise ex
