import mysql.connector

class DatabaseManager:
    def __init__(self, host="localhost", user="root", password="", database="uniwa"):
        self.config = {
            "host": host,
            "user": user,
            "password": password,
            "database": database
        }

    def get_connection(self):
        return mysql.connector.connect(**self.config)

db_manager = DatabaseManager()