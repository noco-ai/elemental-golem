import logging
import argparse
import time
import os
import json
import hashlib
from typing import Dict
import hvac
from application.download import install_skill

pika_logger = logging.getLogger("pika")
pika_logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# local modules
from application.system_info import get_system_info, load_configs, load_enabled_skills
from application.amqp import connect_to_amqp, become_consumer, bind_queue_to_exchange
from application.amqp import create_exchange, create_queue, send_message_to_exchange
from application.thread import start_worker_threads, stop_worker_thread, get_worker_threads, stop_all_threads, update_thread_configuration, stop_thread_generation

# checks what models are installed on the system
def check_data_directories(all_models, all_repos):
    
    # make sure dirs are present
    os.makedirs("data/models", exist_ok=True)    
    os.makedirs("data/loras", exist_ok=True)    
    os.makedirs("data/repos", exist_ok=True)    

    # list of installed models
    available_models = []    
    downloading_models = []
    for model_data in all_models:
        model_name = model_data["path"]
        model_directory = f"data/models/{model_name}"
        lock_file_path = f'data/models/{model_data["lock_file"]}'

        if os.path.exists(lock_file_path):
            downloading_models.append(model_name)
        elif os.path.exists(model_directory):
            available_models.append(model_name)

    # list of insalled repors
    available_repos = []
    for repo_name in all_repos:
        repo_directory = f"data/repos/{repo_name}"

        if os.path.exists(repo_directory):
            available_repos.append(repo_name)

    return available_models, available_repos, downloading_models

def add_skill(new_skill: dict, server_id: str):

    # load existing skills
    enabled_skills = load_enabled_skills(server_id)

    # flatten the list of lists into a single list
    flattened_skills = [skill for sublist in enabled_skills.values() for skill in sublist]

    # expected keys
    expected_keys = {"routing_key": str, "device": str, "use_precision": str}

    # check if new_skill is valid
    if (set(new_skill.keys()) == set(expected_keys.keys()) and
        all(isinstance(new_skill[key], expected_keys[key]) for key in expected_keys)):
        # add new skill to the list
        flattened_skills.append(new_skill)        

        # save back to file
        with open(f'data/{server_id}_skills.json', 'w') as f:
            json.dump(flattened_skills, f)
    else:
        logger.info(f"invalid skill data: {new_skill}")

def remove_skill(skill_to_remove: dict, server_id: str):
    # load existing skills
    enabled_skills = load_enabled_skills(server_id)

    # flatten the list of lists into a single list
    flattened_skills = [skill for sublist in enabled_skills.values() for skill in sublist]

    # iterate over the skills to find the first match and remove it
    for i, skill in enumerate(flattened_skills):
        if skill == skill_to_remove:
            del flattened_skills[i]
            break

    # save back to file
    with open(f'data/{server_id}_skills.json', 'w') as f:
        json.dump(flattened_skills, f)

def fatal_error(error):
    logger.error(error)
    time.sleep(3)
    os._exit(0)

def install_custom_skill(skill_details, server_id):
    data = []    
    filename = f"data/{server_id}_custom.json"
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise TypeError("The file must contain a JSON array.")
            
    data.append(skill_details)
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def connect_to_vault(vault_url, vault_token_file, vault_root, max_retries=10, sleep_time=8):
    for retry in range(max_retries):
        try:
            # Read the Vault token from file
            if os.path.isfile(args.vault_token_file):
                
                # Read the Vault token from file
                with open(vault_token_file, 'r') as file:
                    vault_token = file.read().strip()

                logger.info(f"vault connecting to {vault_url}, vhost: {vault_root}")
                vault_client = hvac.Client(url=vault_url, token=vault_token, namespace=vault_root)
                vault_connected = vault_client.is_authenticated()
                if vault_connected:                    

                    # give a bit of time for amqp to write its creds
                    vault_data_resp = vault_client.read(path=f'{vault_root}/data/core/amqp')
                    if vault_data_resp == None:
                        raise ValueError('invalid response from vault server')                    
                    vault_data = vault_data_resp['data']['data']                    

                    # Check if all required fields are present
                    required_fields = ['host', 'username', 'password', 'vhost']
                    if not all(field in vault_data for field in required_fields):
                        missing_fields = [field for field in required_fields if field not in vault_data]
                        raise ValueError(missing_fields)
                    
                    logger.info('successfully connected to vault server')
                    return vault_client, vault_data
            else:
                logger.info(f"waiting for token file creation")
        except Exception as e:
            pass
        
        time.sleep(sleep_time)            
        logger.info(f"retrying connection to vault server. attempt {retry+1}/{max_retries}")

    # If connection is not successful after max_retries
    fatal_error('unable to connect to vault server after multiple attempts.')    

if __name__ == "__main__":
    logger.info("starting elemental golem")    

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Vault creds')
    parser.add_argument('--server-id', required=True, help='Unique server ID')
    parser.add_argument('--vault-host', required=True, help='Vault server host address')
    parser.add_argument('--vault-token-file', help='Path to the Vault token file', default='./vault-token')
    parser.add_argument('--vault-root', help='Root path in the Vault server', default='spellbook')
    parser.add_argument('--shared-models', required=False, help='Show be set to true is the data/ folder is shared between golem instances or in a docker container.', default=False, type=bool)
    args = parser.parse_args()

    vault_client, vault_data = connect_to_vault(args.vault_host, args.vault_token_file, args.vault_root)        

    # connect to amqp
    amqp_params = {
        'amqp_ip': vault_data['host'],
        'amqp_user': vault_data['username'],
        'amqp_password': vault_data['password'],
        'amqp_vhost': vault_data['vhost']
    }
    server_name = args.server_id
    server_id = 'golem_' + hashlib.sha256(server_name.encode()).hexdigest()[:10]

    # load config files
    all_skills, all_configs, all_models, all_repos, script_map, loaded_handlers = load_configs('modules', vault_client, args.vault_root, server_id)        

    # load enabled models json tp dict
    enabled_skills_dict = load_enabled_skills(server_id)

    # start threads
    start_worker_threads(all_skills, enabled_skills_dict, amqp_params, script_map, server_id)    

    # connect to rabbit mq
    amqp_connected, amqp_connection, amqp_channel = connect_to_amqp(**amqp_params)        
    if amqp_connected == False:
        fatal_error('unable to connect to amqp server')        
    
    # create dead letter exchange and queue
    create_exchange(amqp_channel, 'deadletter')
    flx_queue = create_queue(channel=amqp_channel, queue_name='deadletters')
    bind_queue_to_exchange(amqp_channel, 'deadletters', 'deadletter')    

    # create exchange and queue for this server
    create_exchange(amqp_channel, 'golem')
    create_exchange(amqp_channel, 'golem_broadcast', 'fanout')
    create_queue(channel=amqp_channel, queue_name=server_id, is_auto_delete=True, dlx="deadletter")
    bind_queue_to_exchange(amqp_channel, server_id, 'golem')
    bind_queue_to_exchange(amqp_channel, server_id, 'golem_broadcast')

    # start all the pipe threads
    create_exchange(amqp_channel, 'golem_skill')

    # define server call back for answering messages
    def server_callback(ch, method, properties, body):
        global all_skills, all_configs, all_models, all_repos, script_map, loaded_handlers
        
        if "command" not in properties.headers or "return_routing_key" not in properties.headers or "return_exchange" not in properties.headers:
            logger.info("command or return routing not found in header. command, return_route_key, and return_exchange are required headers")
            amqp_channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
            return
        
        logger.info(f"incoming command {properties.headers['command']}")
        try:                        
            headers = {}
            command = properties.headers.get('command')
            return_key = properties.headers.get('return_routing_key')
            return_exchange = properties.headers.get('return_exchange')

            for key, value in properties.headers.items():
                # Exclude return_exchange and return_routing_key
                if key not in ['return_exchange', 'return_routing_key', 'x-delay']:
                    headers[key] = value

            if command == "system_info":                                
                installed_models, installed_repos, downloading_models = check_data_directories(all_models, all_repos)
                # get list of installed models
                system_info = get_system_info(server_id)
                system_info["server_id"] = server_id
                system_info["server_label"] = server_id.replace("_", "-")
                system_info["installed_models"] = installed_models        
                system_info["downloading_models"] = downloading_models        
                system_info["installed_repository"] = installed_repos        
                system_info["handlers"] = loaded_handlers
                # protect secrets from the UI           
                stripped_skills = [{k: v for k, v in skill.items() if k != "secrets"} for skill in all_skills]
                system_info["installed_skills"] = stripped_skills

                running_skills = []
                system_info["status"] = "ONLINE"
                worker_threads = get_worker_threads()
                for thread in worker_threads:
                    thread_status = thread["thread_status"].raw.decode().rstrip('\0')
                    if thread_status != "ONLINE":
                        system_info["status"] = "STARTING"

                    running_skills.extend([{"device":thread["device"], "routing_key": thread["routing_key"], 
                                            "ram": thread["ram"] * 1000000, "use_precision": thread["use_precision"], 
                                            "thread_status": thread_status }])

                system_info["running_skills"] = running_skills
                send_message_to_exchange(amqp_channel, return_exchange, return_key, json.dumps(system_info).encode(), headers)
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)
                return
            elif command == "run_skill":
                skill_details = json.loads(body)
                add_skill(skill_details, server_id)                
                run_map = {skill_details["routing_key"]: [skill_details]}
                start_worker_threads(all_skills, run_map, amqp_params, script_map, server_id)
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)
                return
            elif command == "stop_skill":
                skill_details = json.loads(body)
                remove_skill(skill_details, server_id)
                stop_worker_thread(skill_details, amqp_channel)
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)
                return
            elif command == "install_skill":
                skill_details = json.loads(body)
                install_skill(all_skills, skill_details, args.shared_models, server_id, amqp_channel)
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info('skill installing 🚀')
                return
            elif command == "custom_skill":
                skill_details = json.loads(body)                
                install_custom_skill(skill_details, server_id)                                
                all_skills, all_configs, all_models, all_repos, script_map, loaded_handlers = load_configs('modules', vault_client, args.vault_root, server_id)
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)                
                logger.info('custom skill installed 🚀')                
                return
            elif command == "stop_generation":                
                stop_details = json.loads(body)
                stop_thread_generation(stop_details)
                logger.info(stop_details)
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info('generation stopped 🛑')
                return            
            elif command == "update_configuration":
                details = json.loads(body)
                vault_data = update_thread_configuration(args.vault_root, vault_client, details["vault_path"])
                for skill in all_skills:
                    if "configuration_template" in skill and "vault_path" in skill["configuration_template"] and skill["configuration_template"]["vault_path"] == details["vault_path"]:
                        current_config = skill["configuration"]
                        merged_config = {**current_config, **vault_data}
                        skill["configuration"] = merged_config
                amqp_channel.basic_ack(delivery_tag=method.delivery_tag)
                logger.info('configuration updated 🔧')
                return            
            
        except Exception as e:
            logger.error("an error occurred:", e)            
            amqp_channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
        
        logger.info(f"command {properties.headers['command']} not found")
        amqp_channel.basic_reject(delivery_tag=method.delivery_tag, requeue=False)        

    become_consumer(amqp_channel, server_id, server_callback)