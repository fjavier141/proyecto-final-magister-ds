import os

from dotenv import load_dotenv

#Ajusta segun tu entorno
env_path = ".env"
load_dotenv(dotenv_path=env_path)

USE_SAVED_HYPERPARAMS = True  #pon False si quieres usar siempre los estándar
HYPERPARAMS_DIR = "./data/output/hyperparams"

USER = os.getenv("POSTGRES_USER")
PASSWORD = os.getenv("POSTGRES_PASSWORD")
HOST = os.getenv("POSTGRES_HOST")
PORT = os.getenv("POSTGRES_PORT", "5432")
DB = os.getenv("POSTGRES_DB")

TEST_PERIOD = int(os.getenv("TEST_PERIOD"))  # Formato AAAAMM, actualmente configurado como : 202412
CATEGORY = os.getenv("CATEGORY")  # 'cervezas' o 'analcoholicos', manejar desde .env la categoría que se usará.
CHANNEL = os.getenv("CHANNEL")
RANDOM_STATE = os.getenv("RANDOM_STATE")

DICT_MIX = {
    "cervezas": ["masivo"],
    "analcoholicos": ["gaseosas", "minerales"]
}

SEGMENTS = ["AL", "BO", "AP", "KI", "BA", "EE", "ES", "FF", "FU", "CD", "IE", "DI", "RT", "RE", "BC", "RC", "GI",
            "FC", "FS", "RD"]

SEGMENTS_CHANNEL = {
    'compra': ["AL", "BO"],
    'consumo': ["AP", "KI", "BA", "EE", "ES", "FF", "FU", "CD", "IE", "DI", "RT", "RE", "BC", "RC", "GI",
            "FC", "FS", "RD"],
    'all': ["AL", "BO", "AP", "KI", "BA", "EE", "ES", "FF", "FU", "CD", "IE", "DI", "RT", "RE", "BC", "RC", "GI",
            "FC", "FS", "RD"]
}