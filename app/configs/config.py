# app/config/config.py
import os
from dotenv import load_dotenv

# Load env file based on ENV variable
ENV = os.getenv("ENV", "dev")  # default dev
load_dotenv(f"app/{ENV}.env")

# --- Database Configuration ---
# FIXED: Providing a fallback value ('3306') prevents os.getenv from returning None.
# If DB_PORT is not found in the environment, it uses '5432'.
DB_PORT = int(os.getenv("DB_PORT", "3306"))

# You should apply this same pattern to other mandatory variables
DB_HOST = os.getenv("DB_HOST", "dummy_host")
DB_USER = os.getenv("DB_USER", "dummy_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "dummy_pwd")
DB_NAME = os.getenv("DB_NAME", "dummy_learning")
FILE_ENV = os.getenv("FILE_ENV", "dummy_file")

# --- Application Configuration ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

print("==== Loaded DB Config ====")
print("DB_HOST:", DB_HOST)
print("DB_PORT:", DB_PORT)
print("DB_USER:", DB_USER)
print("DB_PASSWORD:", DB_PASSWORD)
print("DB_NAME:", DB_NAME)
print("FILE_ENV:", FILE_ENV)
print("==========================")
