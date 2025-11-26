from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import hashlib
import secrets

class UserManager:
    def __init__(self, db_name='user_management', collection_name='users', host='localhost', port=27017):
        try:
            self.client = MongoClient(host, port)
            self.client.admin.command('ping')
            self.db = self.client[db_name]
            self.users_collection = self.db[collection_name]
            
            self.users_collection.create_index('username', unique=True)
            
            print("Successfully connected to MongoDB")
            
        except ConnectionFailure:
            print("Failed to connect to MongoDB")
            raise
    
    def _hash_password(self, password):
        salt = secrets.token_hex(16)
        salted_password = password + salt
        hashed_password = hashlib.sha256(salted_password.encode()).hexdigest()
        return hashed_password, salt
    
    def _verify_password(self, password, salt, stored_hash):
        salted_password = password + salt
        hashed_password = hashlib.sha256(salted_password.encode()).hexdigest()
        return hashed_password == stored_hash
    
    def registerUser(self, username, password):
        try:
            if self.users_collection.find_one({'username': username}):
                print(f"User '{username}' already exists")
                return False
            
            hashed_password, salt = self._hash_password(password)
            
            user_doc = {
                'username': username,
                'password_hash': hashed_password,
                'salt': salt
            }
            
            result = self.users_collection.insert_one(user_doc)
            
            if result.inserted_id:
                print(f"User '{username}' registered successfully")
                return True
            else:
                print(f"Failed to register user '{username}'")
                return False
                
        except DuplicateKeyError:
            print(f"User '{username}' already exists")
            return False
        except Exception as e:
            print(f"Error registering user: {e}")
            return False
    
    def login(self, username, password):
        try:
            user = self.users_collection.find_one({'username': username})
            
            if not user:
                print(f"User '{username}' not found")
                return False
            
            if self._verify_password(password, user['salt'], user['password_hash']):
                print(f"User '{username}' logged in successfully")
                return True
            else:
                print(f"Invalid password for user '{username}'")
                return False
                
        except Exception as e:
            print(f"Error during login: {e}")
            return False
    
    def close_connection(self):
        if hasattr(self, 'client'):
            self.client.close()
            print("MongoDB connection closed")
    
    def __del__(self):
        self.close_connection()