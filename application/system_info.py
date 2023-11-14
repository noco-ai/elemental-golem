from pynvml import nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates, nvmlShutdown, NVMLError
from typing import Dict
import socket
import psutil
import json
import os
import hvac
import hashlib
import logging

logger = logging.getLogger(__name__)

# load what modules are enabled
def load_enabled_skills(server_id: str) -> dict:
    # Check if the file exists
    if not os.path.exists(f'data/{server_id}_skills.json'):
        logger.info(f"file data/{server_id}_skills.json does not exist")
        return {}

    try:
        with open(f'data/{server_id}_skills.json', 'r') as f:
            enabled_skills = json.load(f)
    except json.JSONDecodeError:
        logger.info(f"invalid json in data/{server_id}_skills.json")
        return {}

    # Prepare an empty dictionary to hold valid skills
    enabled_skills_dict = {}

    # Define the expected keys and their data types
    expected_keys = {"routing_key": str, "device": str, "use_precision": str}

    for item in enabled_skills:
        # Check if item contains all expected keys, their values are of the correct data types,
        # and no additional keys are present
        if (set(item.keys()) == set(expected_keys.keys()) and
            all(isinstance(item[key], expected_keys[key]) for key in expected_keys)):

            if item['routing_key'] not in enabled_skills_dict:
                enabled_skills_dict[item['routing_key']] = []

            enabled_skills_dict[item['routing_key']].extend([item])
        else:
            logger.error(f"tnvalid skill data: {item}")

    return enabled_skills_dict

def load_configs(base_dir, vault_client, vault_root, server_id):
    # Return data
    all_skills = []
    all_configs = {}
    all_models = []
    all_repos = []
    script_map = {}
    loaded_handlers = []

    # load custom skills
    custom_skill_map = {}
    custom_skills = []    
    try:        
        filename = f"data/{server_id}_custom.json"
        with open(filename, 'r') as file:
            custom_skills = json.load(file)
    except FileNotFoundError:
        pass

    for custom_skill in custom_skills:
        golem_module_path = f"modules/{custom_skill['golem_module']}"
        if golem_module_path not in custom_skill_map:
            custom_skill_map[golem_module_path] = []

        custom_skill_map[golem_module_path].append(custom_skill)    

    # Walk through the directory
    for dir_path, dir_names, file_names in os.walk(base_dir):

        # Check each file in the current directory
        for file_name in file_names:

            # If the file is not a golem.json file
            if file_name != "golem.json":
                continue

            # Construct the full path to the file
            full_path = os.path.join(dir_path, file_name)

            # Open the file and load the JSON
            with open(full_path, 'r') as f:
                config = json.load(f)

                # Save the loaded config to the dictionary
                script_path = os.path.join(dir_path, config["script"])
                config["script_path"] = script_path
                all_configs[dir_path] = config

                if "repository" in config:
                    for repo in config["repository"]:
                        all_repos.append(repo["folder"])

                # If the "skills" key exists in the JSON, append its contents to the all_models array
                if "skills" in config:
                    if dir_path in custom_skill_map:
                        config["skills"].extend(custom_skill_map[dir_path])

                    loaded_handlers.append({
                        "unique_key": config.get("unique_key", ""),
                        "label": config.get("label", ""),
                        "description": config.get("description", "")
                    })
                    global_repos = config.get("repository", [])
                    global_configuration = config.get("configuration", {})  # Get the global configuration
                    global_config_dict = {option["name"]: option["default"] for option in global_configuration.get("options", [])}
                    vault_path = global_configuration.get("vault_path", "")                    

                    for skill in config["skills"]:                        
                        vault_data = {}
                        if vault_path:
                            try:                                
                                config_path = f'{vault_root}/data/{vault_path}/{skill["routing_key"]}'
                                vault_data_resp = vault_client.read(path=config_path)
                                vault_data = {} if vault_data_resp == None else vault_data_resp['data']['data']
                            except Exception as e:
                                pass # no need to log just means no override data has been set

                        module_name = dir_path.split("modules/")[1]
                        skill["golem_module"] = module_name
                        skill["raw"] = json.dumps(skill, indent=2)
                        skill["handler_key"] = config.get("unique_key", "")
                        skill_configuration = skill.get("configuration", {})
                        merged_config = {**global_config_dict, **skill_configuration, **vault_data}  # Merge global, skill level and vault configurations
                        skill["configuration"] = merged_config  # Replace the skill configuration with the merged configuration
                        skill["configuration_template"] = global_configuration.copy()
                        skill["repository"] = global_repos.copy()                                    
                        for repo in skill["repository"]:
                            repo["module_path"] = dir_path
                            
                        skill["secrets"] = {}

                        if "vault_path" in skill["configuration_template"]:
                            skill["configuration_template"]["vault_path"] = skill["configuration_template"]["vault_path"] + "/" + skill["routing_key"]                        

                        skill["multi_gpu_support"] = True if "multi_gpu_support" in config and config["multi_gpu_support"] == True else False
                        
                        # protect sensetive data
                        if "options" in skill["configuration_template"]:
                            for option in skill["configuration_template"]["options"]:                                
                                if option["type"] == "secret":
                                    skill["secrets"][option["name"]] = merged_config[option["name"]]
                                    merged_config[option["name"]] = "SECRET"
                        
                        all_skills.append(skill)                        
                        script_map[skill["routing_key"]] = script_path

                        if "model" not in skill:
                            continue

                        for model in skill["model"]:
                            if "files" in model:
                                for file in model["files"]:
                                    model_full_path = os.path.join(model["name"], model["files"][file])
                                    lock_file = hashlib.sha256(model_full_path.encode()).hexdigest()[:10] + ".lock"
                                    all_models.append({"path": model_full_path, "lock_file": lock_file })
                            if "branch" in model:
                                for file in model["branch"]:
                                    model_full_path = os.path.join(model["name"], model["branch"][file])
                                    lock_file = hashlib.sha256(model_full_path.encode()).hexdigest()[:10] + ".lock"
                                    all_models.append({"path": model_full_path, "lock_file": lock_file })
                            else:
                                model_full_path = model["name"]
                                lock_file = hashlib.sha256(model_full_path.encode()).hexdigest()[:10] + ".lock"
                                all_models.append({"path": model_full_path, "lock_file": lock_file })                           

    return all_skills, all_configs, all_models, all_repos, script_map, loaded_handlers

def get_gpu_memory_usage(device_id):
    nvmlInit()

    try:
        device_handle = nvmlDeviceGetHandleByIndex(device_id)
        memory_info = nvmlDeviceGetMemoryInfo(device_handle)
        used_memory = memory_info.used / (1024 * 1024) # Convert to MB
        total_memory = memory_info.total / (1024 * 1024) # Convert to MB
        free_memory = memory_info.free / (1024 * 1024) # Convert to MB
    except NVMLError as error:
        logger.info(f"Failed to get GPU memory usage: {error}")
        used_memory = -1
    finally:
        nvmlShutdown()

    return used_memory, free_memory, total_memory

def get_system_info(server_id):
    # network info
    hostname = socket.gethostname()
    system_info = {
        "server_id": server_id,
        "hostname": hostname
    }

    # RAM information
    mem_info = psutil.virtual_memory()
    system_info["ram"] = {
        "total": mem_info.total,
        "available": mem_info.available,
        "used": mem_info.used,
        "percent_used": mem_info.percent
    }

    # CPU information
    system_info["cpu"] = {
        "count": psutil.cpu_count(),
        "percent_used": psutil.cpu_percent()
    }

    # Hard drive information
    disk_usage = psutil.disk_usage(os.path.abspath(os.sep))
    system_info["hd"] = {
        "total": disk_usage.total,
        "used": disk_usage.used,
        "free": disk_usage.free,
        "percent_used": disk_usage.percent
    }

    # NVIDIA GPU information
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    system_info["gpu"] = []
    gpu_names = {}

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        mem_info = nvmlDeviceGetMemoryInfo(handle)
        utilization = nvmlDeviceGetUtilizationRates(handle)

        # rename gpu if we have more than more of the same type
        if name in gpu_names:
            gpu_name = f"{name} #{gpu_names[name]}"
            gpu_names[name] += 1
        else:
            gpu_name = name
            gpu_names[name] = 2

        system_info["gpu"].append({
            "device": f"cuda:{i}",
            "name": gpu_name,
            "memory_total": mem_info.total,
            "memory_used": mem_info.used,
            "memory_free": mem_info.free,
            "gpu_utilization": utilization.gpu,
            "memory_utilization": utilization.memory
        })

    nvmlShutdown()

    # rename multiple gpus
    for gpu in system_info["gpu"]:
        if gpu["name"] in gpu_names and gpu_names[gpu["name"]] > 2:
            gpu["name"] = f"{gpu['name']} #1" 

    return system_info