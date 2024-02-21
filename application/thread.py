import multiprocessing
import importlib
import torch
import time
import traceback
import os
import json
import hashlib
import threading
from pika import BasicProperties
import ctypes
import logging

from application.amqp import connect_to_amqp, become_consumer, bind_queue_to_exchange, send_message_to_exchange
from application.amqp import create_queue
from application.base_handler import BaseHandler
from application.llm_handler import LlmHandler

logger = logging.getLogger(__name__)

worker_threads = []

def load_class_for_skill(skill_key: str, script_map):
    
    if skill_key not in script_map:
        logger.info(f"{skill_key} does not have a defined script")
        return None    

    modules_root = "modules"
    file_path = script_map[skill_key]    
    rel_path = os.path.relpath(file_path, modules_root).replace("\\", "/")
    module_name = rel_path[:-3].replace("/", ".")    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    handler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler_module)
    
    # Search for a handler class that inherits from BaseHandler
    handler_class = None
    for attr in dir(handler_module):
        obj = getattr(handler_module, attr)                
        
        # Check if it's a class first, and then if it's a direct subclass
        if isinstance(obj, type):
            if issubclass(obj, LlmHandler) and obj != LlmHandler:
                handler_class = obj
                break
            elif issubclass(obj, BaseHandler) and obj not in [BaseHandler, LlmHandler]:
                handler_class = obj
                break

    if handler_class is not None:
        handler_instance = handler_class()
        return handler_instance

    logger.warning(f"no module class found for skill '{skill_key}'")
    return None

# notify front end of updates
def send_ui_update(command, skill_key, server_id, channel):
    outgoing_headers = { 
        "command": command
    }
    outgoing_properties = BasicProperties(headers=outgoing_headers)
    channel.basic_publish(
        exchange="arcane_bridge_broadcast", 
        routing_key="", 
        body=json.dumps({
            "skill_key": skill_key,
            "server_id": server_id
            }), properties=outgoing_properties)
        
def worker_thread(amqp_params, stop_event, stop_generation_event, stop_generation_filter, thread_status, config_event, thread_config, 
                  device_and_status, skill, script_map, server_id):
        
    skill_key = skill["routing_key"]
    short_hash = hashlib.sha256(skill_key.encode()).hexdigest()[:10]
    queue_name = f"skill_{short_hash}"    
    current_skill = load_class_for_skill(skill_key, script_map)
    if current_skill is None:
        logger.warning(f"could not load skill {skill['routing_key']}")
        return
        
    amqp_connected, connection, channel = connect_to_amqp(**amqp_params)    
    if amqp_connected == False:
        return
    
    ## check to make sure mode will work on selected device
    channel.basic_qos(prefetch_count=1)
    device = device_and_status["device"]
    root_device = device.split(":")[0]
    if root_device == "split":
        root_device = "cuda"
        
    if root_device not in skill["available_precision"]:
        logger.warning(f"precision not defined for skill {skill['routing_key']}")
        return

    if device_and_status["use_precision"] not in skill["available_precision"][root_device]:
        logger.warning(f"skill {skill['routing_key']} precision {device_and_status['use_precision']} not supported for device {root_device}")
        return

    # load the models
    model_name = "" if "model" not in skill else skill["model"][0]["name"]
    local_path = f"data/models/{model_name}"
    
    load_error = False
    loaded_skill = current_skill.load(skill, device_and_status, local_path)
    if "error" in loaded_skill and loaded_skill["error"] == True:
        load_error = True

    # try to load the model to device
    if "model" in loaded_skill:        
        try:
            loaded_skill["model"].to(device)
            logger.info(f"skill {skill_key} loaded to {device}")    
        except:        
            try:
                loaded_skill["model"].device = torch.device(device)
                loaded_skill["model"].model.to(device)       
                logger.info(f"skill {skill_key} loaded to {device}")    
            except Exception as e:
                logger.error("error occured while loading model", e)
                load_error = True                

    if load_error == False:
        try:                        
            create_queue(channel=channel, queue_name=queue_name, dlx='deadletter', dlx_queue='deadletters', is_auto_delete=False)
            bind_queue_to_exchange(channel, queue_name, 'golem_skill', skill["routing_key"])
            loaded_skill["secrets"] = skill["secrets"]
            loaded_skill["amqp_channel"] = channel    
            loaded_skill["amqp_connection"] = connection
        except Exception as e:        
            logger.error("failed to bind to queue", e)    
    
    # stop genetation events data
    loaded_skill["stop_generation_event"] = stop_generation_event
    loaded_skill["stop_generation_filter"] = stop_generation_filter
    
    # Set up the callback function to handle incoming requests
    def worker_callback(ch, method, properties, body):
        
        if stop_event.is_set():             
            channel.stop_consuming()            
            channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return

        if not hasattr(properties, 'headers') or properties.headers == None or any(key not in properties.headers for key in ["return_exchange", "return_routing_key", "command"]):
            channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            logger.error('missing required amqp headers')
            return
        
        # give execute access to amqp headers
        errors = []
        result_json = "{}"
        call_success = True        
        loaded_skill["amqp_headers"] = properties.headers        
        result = None

        try:
            start = time.time()           
            body_json = json.loads(body.decode())
            valid, errors = current_skill.validate(body_json)
            
            if valid == False:
                call_success = False
            else: 
                result = current_skill.execute(loaded_skill, body_json)
                logger.info(f"[ {skill_key} ] took {time.time() - start:.3f}s")
                if result is None:
                    call_success = False
                    errors = [f"skill {skill_key} returned no data"]
                else:    
                    result_json = json.dumps(result)        

        except Exception as e:                                    
            logger.error("error occured while executing handler", e)            
            errors = ["An error occured with the skill"]
            call_success = False

        try:
            # copy headers to outgoing message
            outgoing_headers = {}
            for incoming_header in properties.headers:
                if incoming_header in ["x-delay", "return_exchange", "return_routing_key"]:
                    continue
                outgoing_headers[incoming_header] = properties.headers[incoming_header]                

            outgoing_headers["success"] = call_success
            outgoing_headers["errors"] = errors
            outgoing_properties = BasicProperties(headers=outgoing_headers)
            channel.basic_publish(
                exchange=properties.headers['return_exchange'], 
                routing_key=properties.headers['return_routing_key'], 
                body=result_json, properties=outgoing_properties)

            channel.basic_ack(delivery_tag=method.delivery_tag)   
        except Exception as e:           
            logger.error("error occured while publishing message", e)    
        
    # Become a consumer of the queue
    def consume():                
        become_consumer(channel, queue_name, worker_callback)
        send_ui_update("skill_stopped", skill_key, server_id, channel)
        thread_status.raw = bytes('\0' * 24, 'utf-8')
        thread_status.raw = bytes("STOPPED", "utf-8")  
        channel.connection.close()  
        
    thread_status.raw = bytes('\0' * 24, 'utf-8')
    if load_error == False:
        consumer_thread = threading.Thread(target=consume)
        thread_status.raw = bytes("ONLINE", "utf-8")
        send_ui_update("skill_started", skill_key, server_id, channel)
        consumer_thread.start()
    else:        
        thread_status.raw = bytes("ERROR", "utf-8")        

    # endless loop until thread user asks to stop thread
    while True:        
        time.sleep(2)
        thread_string = bytes(thread_status.raw).rstrip(b'\x00').decode("utf-8")
        if thread_string == "STOPPED":
            break
        
        if load_error and thread_string == "STOPPING":
            break

        if config_event.is_set():
            json_string = bytes(thread_config.raw).rstrip(b'\x00').decode("utf-8")
            config_json = json.loads(json_string)
            current_skill.update_config(config_json)
            config_event.clear()            

def get_worker_threads():
    return worker_threads

def stop_all_threads(amqp_channel):
    
    for i, thread in enumerate(worker_threads):
        logger.info(f"stopping thread for {thread['routing_key']}")        
        thread["thread_status"].raw = bytes('\0' * 24, 'utf-8')   
        thread["thread_status"].raw = bytes("STOPPING", "utf-8") 
        thread["stop_event"].set()
        send_message_to_exchange(amqp_channel, "golem_skill", thread["routing_key"], "STOP", None)            
        while True:
            time.sleep(2)
            thread_string = bytes(thread["thread_status"].raw).rstrip(b'\x00').decode("utf-8")
            if thread_string == "STOPPED":
                break

        thread["process"].join()
        del worker_threads[i]

def stop_worker_thread(skill_details, amqp_channel):
    
    for i, thread in enumerate(worker_threads):
        if thread["routing_key"] == skill_details["routing_key"] and thread["device"] == skill_details["device"] and thread["use_precision"] == skill_details["use_precision"]:            
            logger.info(f"stopping thread for {skill_details['routing_key']}")        
            thread["thread_status"].raw = bytes('\0' * 24, 'utf-8')   
            thread["thread_status"].raw = bytes("STOPPING", "utf-8") 
            thread["stop_event"].set()
            send_message_to_exchange(amqp_channel, "golem_skill", skill_details["routing_key"], "STOP", None)            
            while True:
                thread_string = bytes(thread["thread_status"].raw).rstrip(b'\x00').decode("utf-8")
                if thread_string == "STOPPED":
                    break

            thread["process"].join()
            del worker_threads[i]
            return            

def stop_thread_generation(stop_details):
    
    for i, thread in enumerate(worker_threads):
        if thread["routing_key"] == stop_details["routing_key"]:            
            logger.info(f"sending stop generation to {stop_details['routing_key']}")
            thread["stop_generation_filter"].raw = bytes('\0' * 128, 'utf-8')
            thread["stop_generation_filter"].raw = bytes(stop_details["socket_id"], "utf-8")        
            thread["stop_generation_event"].set()
            return
        
def update_thread_configuration(vault_root, vault_client, vault_path):
    
    config_path = f'{vault_root}/data/{vault_path}'
    logger.info(f"updating thread configuration for {config_path}")
    vault_data_resp = vault_client.read(path=config_path)
    vault_data = {} if vault_data_resp == None else vault_data_resp['data']['data']
    path_parts = vault_path.split('/')
    unique_key = path_parts[-1]
    json_dump = json.dumps(vault_data)
    if len(json_dump) >= 4096:
        #logger.error(f"error: configuraation json longer than buffer")
        return {}
    
    for thread in worker_threads:
        if thread["routing_key"] != unique_key:
            continue
        
        thread["thread_config"].raw = bytes('\0' * 4096, 'utf-8') 
        thread["thread_config"].raw = bytes(json_dump, "utf-8")
        thread["config_event"].set()
        
    return vault_data

def start_worker_threads(all_skills, skills_config, amqp_params, script_map, server_id):
    
    # Iterate through pipelines in the config
    for skill in all_skills:
        routing_key = skill["routing_key"]

        # Skip the pipeline if the name is not found in the devices_and_status_dict
        device_and_status = skills_config.get(routing_key)
        if device_and_status is None:
            continue

        for to_device in device_and_status:

            # Create a new process for each consumer
            stop_generation_event = multiprocessing.Event()
            stop_generation_filter = multiprocessing.Array(ctypes.c_char, 128)
            stop_event = multiprocessing.Event()
            thread_status = multiprocessing.Array(ctypes.c_char, 24)
            config_event = multiprocessing.Event()
            thread_config = multiprocessing.Array(ctypes.c_char, 4096)
            thread_status.raw = bytes("STARTING", "utf-8")
            process = multiprocessing.Process(target=worker_thread, args=(amqp_params, stop_event, stop_generation_event, stop_generation_filter, 
                                                                          thread_status, config_event, thread_config, to_device, skill, script_map, server_id))
            process.start()

            device = to_device["device"]
            ram = skill["memory_usage"][to_device["use_precision"]]
            worker_threads.extend([{ "process": process, 
                                    "routing_key": routing_key, "device": device, "ram": 
                                    ram, "use_precision": to_device["use_precision"], "stop_event": stop_event, "stop_generation_event": stop_generation_event,
                                    "stop_generation_filter": stop_generation_filter, "thread_status": thread_status, "config_event": config_event, "thread_config": thread_config}]) 