from urllib.parse import urlparse
from pika import BasicProperties
import hvac
import json
import jsonschema
from jsonschema import validate
from typing import Union, List
import logging

logger = logging.getLogger(__name__)

class BaseHandler:

    def __init__(self):
        self.cached_schemas = {}

    def execute(self, model, request) -> dict:
        raise NotImplementedError("The `execute` method should be implemented in the derived class.")
    
    def validate(self, request) -> dict:
        raise NotImplementedError("The `validate` method should be implemented in the derived class.")

    def load(self, model, model_options) -> dict:
        return {}
    
    def copy_queue_headers(self, incoming_headers, override_command = None):
        # copy amqp headers
        outgoing_headers = {}
        stream_override = None
        for incoming_header in incoming_headers:
            if incoming_header in ["x-delay", "return_exchange", "return_routing_key"]:
                continue
            if incoming_header == "stream_to_override":
                stream_override = incoming_headers[incoming_header]

            outgoing_headers[incoming_header] = incoming_headers[incoming_header]

        stream_to = "prompt_fragment" if stream_override == None else stream_override
        outgoing_headers["command"] = override_command if override_command is not None else stream_to
        return BasicProperties(headers=outgoing_headers)

    def load_schema_file(self, schema_file):
        # Check if schema is in cache
        if schema_file in self.cached_schemas:
            schema = self.cached_schemas[schema_file]
        else:
            # Load the schema at the path
            try: 
                with open(f"schema/{schema_file}.jsonschema", 'r') as file:
                    schema = json.load(file)
            except Exception as e:
                logger.error(e)
                return None
            # Cache the schema
            self.cached_schemas[schema_file] = schema

        return schema
    
    # A dictionary to hold the cached schemas    
    def validate_request(self, json_data: dict, schema_file: str) -> Union[bool, List[str]]:
        
        schema = self.load_schema_file(schema_file)
        if schema is None:
            return False, ["Invalid schema file for handler"]
        
        json_data = self.apply_schema_defaults(json_data, schema_file)        
        try:
            validate(instance=json_data, schema=schema)
        except jsonschema.exceptions.ValidationError as err:
            # If there is a validation error, return a list containing the error message
            logger.warn("validation failed for incoming request")
            return False, [str(err)]
        else:
            # If the data is valid, return True
            return True, []

    def apply_schema_defaults(self, raw_data: dict, schema_file: str) -> dict:
        
        schema = self.load_schema_file(schema_file)
        if schema is None:
            logger.error("could not load schema file")
            return raw_data
        
        # Fill default values
        for property, attributes in schema['properties'].items():
            if "default" in attributes and property not in raw_data:
                raw_data[property] = attributes["default"]

        return raw_data

    
    def check_stop_generation(self, counter, stop_generation_event, stop_generation_filter, socket_id):        
        counter += 1
        if counter >= 5:
            counter = 0
            if stop_generation_event.is_set():
                stop_generation_event.clear()
                if socket_id == None:
                    return False, counter
                
                stop_socket = bytes(stop_generation_filter.raw).rstrip(b'\x00').decode("utf-8")
                if stop_socket == socket_id:
                    return True, counter

        return False, counter
