import os 
from pathlib import Path
import logging 

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s]: %(message)s")


list_of_files = [
    ".github/workflows/.gitkeep",

    "app/__init__.py",
    "app/config.py",
    "app/logging_config.py",
    "app/main.py",
    "app/utils.py",

    "app/services/__init__.py",
    "app/services/cache_service.py",
    "app/services/docling_service.py",
    "app/services/embeddings_service.py",
    "app/services/local_storage.py",
    "app/services/query_cache_service.py",
    "app/services/rag_service.py",
    "app/services/router_service.py",
    "app/services/s3_storage.py",
    "app/services/sql_service.py",
    "app/services/storage_backend.py",
    "app/services/vector_service.py",

    "data/sql/schema.sql",
    "data/generate_sample_data.py",

    "tests/test_queries.json",
    "tests/test_storage_backends.py",

    ".dockerignore",
    "evaluate.py",
    "lambda_handler.py",

    "docker-compose.yml",
    "Dockerfile",
    "Dockerfile.lambda",
    "Dockerfile.lambda.with-tesseract",

    "supabase_con_test.py",
    
    ".env.example",
    ".env",
]

for file_path in list_of_files:
    file_path =  Path(file_path)
    file_dir,file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir,exist_ok=True)
        logging.info(f"Creating directory: {file_dir} for file: {file_name}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path,"w") as f:
            pass
            logging.info(f"Creating an empty file: {file_path}")
    
    else:
        logging.info(f"{file_name} already exists")
