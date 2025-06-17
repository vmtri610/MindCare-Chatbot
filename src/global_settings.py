import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_FILE = "data/cache/pipeline_cache.json"
CONVERSATION_FILE = "data/cache/chat_history.json"
STORAGE_PATH = "data/ingestion_storage/"
FILES_PATH = [
    os.path.join(
        PROJECT_ROOT,
        "data",
        "ingestion_storage",
        "dsm-5-cac-tieu-chuan-chan-doan.pdf"
    )
]
INDEX_STORAGE = "data/index_storage"
SCORES_FILE = "data/user_storage/scores.json"
USERS_FILE = "data/user_storage/users.yaml"
