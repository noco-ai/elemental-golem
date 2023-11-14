import os
import logging
import shutil
import requests
import multiprocessing
import hashlib
from huggingface_hub import snapshot_download, hf_hub_download
from application.thread import send_ui_update

logger = logging.getLogger(__name__)

BUFFER_SIZE = 64 * 1024 * 1024  # 64 MB

# Usage
def install_skill(all_skills, install_skill_data, shared_models, server_id, channel):
    # Create a list to hold all the processes
    processes = []
    for skill in all_skills:
        if skill["routing_key"] != install_skill_data["routing_key"]:
            continue

        if "model" in skill:
            for model in skill["model"]:
                process = multiprocessing.Process(target=download_model, args=(model, install_skill_data, shared_models, server_id, channel))
                processes.append(process)
                process.start()
        
        if "repository" in skill:
            for repo in skill["repository"]:
                # Create and start a new process for each download
                process = multiprocessing.Process(target=download_repo, args=(repo["url"], repo["folder"], repo["module_path"]))
                processes.append(process)
                process.start()

def download_repo(url, repo_folder, target_folder):    
    repo_folder = f'data/repos/{repo_folder}'
    
    if os.path.exists(repo_folder) and os.path.isdir(repo_folder):
        os.system(f"cd {repo_folder} && git pull")
    else:
        # Create the directory if it doesn't exist
        os.makedirs(repo_folder, exist_ok=True)
        os.system(f"git clone {url} {repo_folder}")

    logger.info(f"done downloading repository {repo_folder}")
    
    # Make sure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Copy all files from repo_folder to target_folder
    for file_name in os.listdir(repo_folder):
        source = os.path.join(repo_folder, file_name)
        destination = os.path.join(target_folder, file_name)

        if os.path.isfile(source):
            shutil.copy2(source, destination)
        elif os.path.isdir(source):
            shutil.copytree(source, destination, dirs_exist_ok=True)

    logger.info(f"done copying repository files to {target_folder}")

def combine_files(base_file, split_files):
    
    logger.info(f"opening {base_file} to combing splits")
    with open(base_file, 'wb') as dest_file:
        for split_name in split_files:
            filename = base_file + split_name
            logger.info(f"combining split file {filename}")
            with open(filename, 'rb') as src_file:
                while True:
                    chunk = src_file.read(BUFFER_SIZE)
                    if not chunk:
                        break
                    dest_file.write(chunk)
            
            logger.info(f"cleaning up split file {split_name}")
            os.remove(filename)
    logging.info("done combining split files")

def download_model(model, install_skill, shared_models, server_id, channel):
    
    name = model["name"] 
    provider = model["provider"]
    is_branch = False
    single_file = False
    model_full_path = model["name"]
    if "files" in model and install_skill["precision"] in model["files"]:
        model_full_path = os.path.join(model["name"], model["files"][install_skill["precision"]])
        single_file = True
    elif "branch" in model and install_skill["precision"] in model["branch"]:
        model_full_path = os.path.join(model["name"], model["branch"][install_skill["precision"]])
        is_branch = True
    
    lock_file_path = "data/models/" + hashlib.sha256(model_full_path.encode()).hexdigest()[:10] + ".lock"

    # Check if a download is already in progress for the given model name
    if os.path.exists(lock_file_path):
        logger.info(f"download already in progress for model: {name}")
        return       

    # Create a lock file to signal that a download is in progress
    with open(lock_file_path, 'w') as lock_file:
        lock_file.write("download in progress")
    
    logger.info(f"downloading skill model {name}")

    try:        
        if provider == 'huggingface':
            os.makedirs(f'data/models/{name}', exist_ok=True)
            use_symlinks = False if shared_models else "auto"
            cache_dir = "data/cache" if shared_models else None
            download_args = {
                "repo_id": name,
                "local_dir": f'data/models/{model_full_path}'                
            }

            # Conditionally adding arguments to the dictionary if they are not None
            if cache_dir is not None:
                download_args["cache_dir"] = cache_dir

            if use_symlinks is not None:
                download_args["local_dir_use_symlinks"] = use_symlinks

            if single_file:
                if "split" in model and install_skill["precision"] in model["split"]:
                    for split_name in model["split"][install_skill["precision"]]:
                        download_args["filename"] = model["files"][install_skill["precision"]] + split_name
                        download_args["local_dir"] = f'data/models/{name}'                
                        logger.info(f'downlading split file {model["files"][install_skill["precision"]]}{split_name} from hf hub, shared: {shared_models}')
                        hf_hub_download(**download_args)
                    
                    base_file = os.path.join('data', 'models', model_full_path)            
                    combine_files(base_file, model["split"][install_skill["precision"]])                    
                else:
                    logger.info(f'downlading single file {model["files"][install_skill["precision"]]} from hf hub, shared: {shared_models}')
                    download_args["local_dir"] = f'data/models/{name}'
                    download_args["filename"] = model["files"][install_skill["precision"]]
                    hf_hub_download(**download_args)
            elif is_branch:
                logger.info(f'downlading branch {model_full_path} from hf hub, shared: {shared_models}')
                os.makedirs(f'data/models/{model_full_path}', exist_ok=True)
                download_args["revision"] = model["branch"][install_skill["precision"]]
                snapshot_download(**download_args)
            else:
                logger.info(f'downlading repo {name} from hf hub, shared: {shared_models}')
                snapshot_download(**download_args)
        elif provider == 'civitai':
            url = "" if "url" not in model else model["url"]
            if 'api' not in url:
                logger.error("invalid url provided for civit.ai")
                return

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            base_name = name.rsplit('/', 1)[0]
            base_dir = f'data/models/{base_name}'
            os.makedirs(base_dir, exist_ok=True)

            logger.info(f"downloading model from {url}, please wait...")
            response = requests.get(url, headers=headers, stream=True)
            with open(os.path.join(base_dir, name.split('/')[-1]), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("model downloaded successfully!")
        else:
            logger.error(f"{provider} is not supported")

    finally:
        # Remove the lock file
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

        send_ui_update("skill_downloaded", name, server_id, channel)
        logger.info(f"finished downloading skill model {name}")        