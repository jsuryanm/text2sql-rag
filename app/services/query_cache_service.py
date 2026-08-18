import hashlib 
import json 
import logging 
from pathlib import Path 
from typing import List, Dict, Any, Optional 
import numpy as np 

from app.services.storage_backend import StorageBackend 
from app.services.local_storage import LocalStorageBackend 
# from app.services.s3_storage import S3Storage
