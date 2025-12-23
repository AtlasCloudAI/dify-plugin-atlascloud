# Atlas Cloud Dify Plugin Privacy Policy

This privacy policy explains how the Atlas Cloud Dify Plugin collects, uses, and processes user data. Please read this policy carefully before using this plugin.

## Data Collection

This plugin collects and processes the following types of data:

### Configuration Information

- **AtlasCloud API Key**
  - **Purpose**: Used for authentication with the atlascloud.ai service to access AI models.
  - **Data Type**: Secret credential string
  - **Storage Location**: Stored only in your Dify instance configuration. Will not be sent to any third-party services except atlascloud.ai.
  - **Retention**: Stored until you delete or modify the configuration.

### User Input Data

- **User Messages and Prompts**
  - **Purpose**: Sent to AtlasCloud AI models to generate responses.
  - **Data Type**: Text content, may include user queries, conversation context, and multimodal content (text, images).
  - **Processing**: Data is transmitted to AtlasCloud's API endpoints for inference.
  - **Storage**: The plugin itself does not persistently store user messages. Storage behavior depends on your Dify instance's configuration.

### AI Response Data

- **Generated AI Responses**
  - **Purpose**: Returned to users as AI-generated content.
  - **Data Type**: Text responses, tool calls, and metadata.
  - **Storage**: Responses are processed transiently and returned to the Dify application. Persistent storage is handled by the Dify platform, not this plugin.

### Technical Metadata

- **API Request Metadata**
  - **Purpose**: Required for API communication and error handling.
  - **Data Type**: Model names, token usage statistics, request timestamps, error logs.
  - **Storage**: Temporary logs for debugging and operational purposes.

## Data Usage

The data collected by this plugin is used solely for the following purposes:

1. **Authentication**: Validating your AtlasCloud API key to establish secure connections
2. **AI Model Interaction**: Sending user prompts to AtlasCloud AI models and returning generated responses
3. **Error Handling**: Logging errors to diagnose and resolve technical issues
4. **Usage Tracking**: Monitoring token usage for billing and performance optimization

### Third-Party Services

This plugin interacts with the following third-party service:

**AtlasCloud (atlascloud.ai)**

- **API Endpoint**: `https://api.atlascloud.ai/v1`
- **Shared Data**:
  - API Key (for authentication)
  - User messages and prompts
  - Conversation context and tools
  - Model configuration parameters
- **Purpose**: To access AI models provided by AtlasCloud
- **Privacy Policy**: [AtlasCloud Privacy Policy](https://atlascloud.ai/privacy) (Please refer to AtlasCloud's official privacy policy for details on how they handle your data)

**OpenAI Python Library**

- **Purpose**: Used as the HTTP client for API communication
- **Data Handling**: This is a client library that facilitates API calls. It does not store or process data independently.

## Data Security

This plugin implements the following security measures:

1. **API Key Protection**: API keys are stored as encrypted secrets in your Dify instance
2. **HTTPS Encryption**: All communications with AtlasCloud API use HTTPS/TLS encryption
3. **No Local Storage**: The plugin does not write user data to local files or databases
4. **Minimal Data Retention**: Data is processed in-memory and only retained as long as necessary for request processing
5. **Secure Transmission**: All data transmitted to AtlasCloud is encrypted in transit

## Data Sharing

This plugin **does not** share your data with any third parties other than:

- **AtlasCloud**: Required for AI model inference (as detailed above)
- **Required technical dependencies**: Standard libraries needed for plugin operation

We do not:

- Sell your data to advertisers
- Share data with analytics services
- Use data for training purposes
- Share data with affiliates or partners

## User Rights

As a user of this plugin, you have the right to:

1. **Transparency**: Understand exactly how your data is processed
2. **Control**: Delete or modify your API key configuration at any time
3. **Deletion**: Remove the plugin entirely to stop all data processing

## Children's Privacy

This plugin is not intended for use by individuals under the age of 13. If you are under 13, please do not use this plugin or provide any personal information.

## Changes to This Privacy Policy

We may update this privacy policy from time to time. Significant changes will be reflected in version updates to the plugin. We recommend reviewing this policy periodically.

## Contact Information

If you have questions or concerns about this privacy policy, please contact:

- **Plugin Author**: AtlasCloud
- **Repository**: <https://github.com/AtlasCloudAI/dify-plugin-atlascloud>
- **AtlasCloud Support**: <support@atlascloud.ai>

---

**Last Updated**: December 2025
**Version**: 0.0.1

---
