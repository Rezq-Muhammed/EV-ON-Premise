# 01_config/load_config.py

import yaml
import os
from dotenv import load_dotenv

def load_config(config_path="01_config/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_env(env_path=".env"):
    load_dotenv(env_path)
    env_vars = {k: v for k, v in os.environ.items()}
    return env_vars

if __name__ == "__main__":
    config = load_config()
    env = load_env()
    print("Config:", config)
    print("Env:", env.get("OPENAI_API_KEY"))
