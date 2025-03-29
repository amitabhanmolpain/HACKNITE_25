from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

url = "mongodb+srv://adwaitaayush:2gqqzpwRCrfcfmzr@cluster0.1443dcz.mongodb.net/"

client = MongoClient(url, server_api=ServerApi('1'))

db=client.Hacknite_db
collection=db["Users"]