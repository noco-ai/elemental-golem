# Elemental Golem

Elemental Golem is a project that defines and serves AI models using a modular system with a `golem.json` configuration file and a handler that implements the call and response from the model utilizing AMQP as the message broker. It is the backend used by Arcane Bridge and Spell Book for interating with AI models based off Pytorch and similar libraries. It
currently focuses soley on inference tasks.

## Stack Architecture

![Software stack diagram](https://github.com/noco-ai/spellbook-docker/blob/master/stack.png)

## Dependencies

- Hashicorp Vault >= 1.1
- RabbitMQ >= 3.6.10

### Required Vault Keys

In order to function Elemental Golem need to connect to a Vault server to retervice secrets and configuration data.
The following information needs to be stored in Vault for Element Golem to start.

### **core/amqp**

```json
{
  "host": "127.0.0.1",
  "password": "securepass",
  "username": "spellbook-user",
  "vhost": "spellbook"
}
```

## Install Guide

### Docker Install

See https://github.com/noco-ai/spellbook-docker for installing the entire Spell Book stack with Docker Compose.

### CLI Parameters and Server Commands

Elemental Golem provides several CLI commands for controlling the software. Below is a detailed explanation of them.

### **Command-line Interface (CLI) Parameters**:

- `--server-id`: A unique identifier for the server. Required parameter.
- `--vault-host`: The address of the Vault server host. Required parameter.
- `--vault-token-file`: The path to the file containing the Vault token. Defaults to './vault-token' if not specified.
- `--vault-root`: The root path in the Vault server. Defaults to 'arcane-bridge' if not specified.
- `--shared-models`: If set to true all downloads for HuggingFace will be do to the data/cache/ folder. This is useful for shared drives and docker.
- `--amqp-ip`: Overrides the IP stored in Vault for connecting to AMQP server. Useful if running instances of Elemental Golem on additional servers when primary node is running stack using Docker compose.

### **Commands to Control the Server over AMQP**:

- `system_info`: Returns details about the system, such as server ID, system status (ONLINE or STARTING), installed skills, and running skills.
- `run_skill`: Adds and runs a skill on the server based on the `skill_details` provided in the message body.
- `stop_skill`: Stops a running skill based on the `skill_details` provided in the message body.
- `install_skill`: Installs a skill on the server based on the `skill_details` provided in the message body.
- `stop_generation`: Stops generation on a particular thread based on the `stop_details` provided in the message body.
- `update_configuration`: Updates the configuration of the system based on the details provided in the message body.

Each command request should contain a `command`, `return_routing_key`, `return_exchange` in the message `headers`. Based on the command executed, appropriate responses are provided through the `AMQP` channel.

> Note: It is crucial to reject the message correctly if any error occurs during command execution to prevent the message broker from requeueing the message.

### LLM Payload Validation

The LLM handlers check the AMQP payload for the following data:

- **max_new_tokens** (Number, Required): Your desired maximum tokens generated.
- **top_p** (Number, Required): Your desired randomness in response (0.0 to 1.0).
- **temperature** (Number, Required): Your desired "temperature" of output (0.0 to 1.0).
- **stream** (Boolean, Required): If set to true, it signals to stream the output.
- **debug** (Boolean, Optional): Signals to enable the debug mode. If enabled model output will be streamed to the console.
- **stop_key** (String, Optional): The key string to stop generation.
- **lora** (String, Optional): Specifies a lora to use with the request. Only ExLlama support if included at this point.
- **ai_role** (String, Optional): Specifies role of AI in conversations.
- **user_role** (String, Optional): Specifies role of user in conversations.
- **start_response** (String, Optional): Specifies the response to start with.
- **raw** (String, Optional): Raw content to use for generating the prompt.
- **messages** (Array, Required): An array of message objects with these properties:
  - **role** (String, Required): Role in the message.
  - **content** (String, Required): Content of the message.

## golem.json

The golem.json file defines the handlers and models/skills available for loading and inference. Here is a high level overview of the the fields found in the file.
The best reference for this at the moment is to look in the modules/noco-ai/... to file and look at a handler that is implements the handler for similar type of model.
If your model uses transformers of 🤗 pipelines you can add a new definition for the model to an exisiting handler.

The configuration for Elemental Golem is stored in a JSON file. Below is a breakdown of each field in the JSON file:

- `label`: Name of the module.
- `description`: Purpose of the module, what the skill does.
- `script`: Python script to use in running the project.
- `multi_gpu_support`: Boolean indicating multi-gpu support.
- `repository`: Stores information about the code repository.
  - `url`: URL to the project repository.
  - `folder`: Specific directory within the repository URL.
- `skills`: An array containing model definitions. Each model has its properties:
  - `label`: A readable name for the model.
  - `routing_key`: Routing key for the message broker.
  - `use`: Use case(s) for this skill.
  - `available_precision`: Array of information of what devies and percision the skill can be loaded at.
  - `memory_usage`: The memory capacity requirement of the model.
  - `model`: A model with a `name` and `provider`.
  - `shortcut`: A symbol representing the model, this allows LLM models to be accessed via the Spellbook UI directly.
  - `configuration`: Model-specific configurations, like `max_seq_len`, `user_role`, `ai_role`, `stop_on`, `system_messages`.
- `configuration`: Module-wide configuration involving secrets management and system-specific parameters.
  - `vault_path`: The path to secrets storage for sensitive data like API tokens.
  - `options`: An array of global options.

### Model Configuration

Each model/skill can define configurtion information that is available to the handler. If these have keys that match the global configuration keys
for the module they are merged with the user set values overriding the defaults. Here is an example of configuration values a LLM handler expects.

- `max_seq_len`: Specifies the maximum sequence length for model input.
- `user_role`: Define the user's assumed role in the interaction.
- `ai_role`: Defines the AI's assumed role in the interaction.
- `stop_on`: The signals that, when received, will trigger the model to stop the execution.
- `system_message`: Describes the nature of interaction between a user and the AI.

### Global Configuration

The global configuration is read by the frontend which allows the user to override the system default. What configuration options will vary by the type
of handler. Module-wide configuration options include.

- `vault_path`: Secure storage path for sensitive data like API keys.
- `options`: An array of global parameters each with:
  - `label`: A readable field name displayed in a settings UI.
  - `name`: Identifier for the option/field.
  - `editable`: Boolean determining if the user can manually edit the value.
  - `type`: Data type of the parameter.
  - `default`: The default value if none is provided.

### Repository

Some modules require that another repo is installed to allow for a skill handler to work correctly. These are defines at a global level for the handler.

- `url`: URL to the project repository.
- `folder`: The path to the folder within the repository.

## handler.py

The handler is a Python class inheriting from `BaseHandler` or `LlmHandler` that is responsible for handling messages. Each handler must implement the following functions:

- `__init__`: Initialize the handler.
- `validate`: Validates a request. It should return a boolean indicating whether the request is valid and a list of errors (if any).
- `execute`: Executes the model. It receives the model and request. This method is responsible for getting the request data, making the API call, and returning API response.
- `load`: Loads the model. Receives three parameters: the model, model options, and the local path to the model. Be sure to set up the API key using `model["secrets"]["token"]`.

```python
class ExampleHandler(BaseHandler):
    def __init__(self):
        super().__init__()

    def validate(self, request):
        return self.validate_fields(request, [("text", len)])

    def execute(self, model, request):
        #...

    def load(self, model, model_options, local_path):
        openai.api_key = model["secrets"]["token"]
        return {"model_name": model["configuration"]["model"]}
```

Remember to replace `#...` in `execute` with the correct implementation that fits your scenario.
The response must return `{ "content": response}` where `response` is the content you wish to send back.

_Configuration, requests, and responses vary based on how the handler is implemented._
