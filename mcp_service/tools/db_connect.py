import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

def connect_to_postgres():
    db_uri = os.getenv("POSTGRES_URI")

    if not db_uri:
        raise ValueError("POSTGRES_URI is not set")

    return SQLDatabase.from_uri(db_uri)

# 34.235.156.160
# postgres:
#     image: postgres:16
#     container_name: postgres
#     ports:
#       - "5432:5432"
#     environment:
#       POSTGRES_USER: nQFHfn5sq
#       POSTGRES_PASSWORD: 8iZSJDPi6ccSk7IGnEYM2kcU
#       POSTGRES_DB: app
#     volumes:
#       - postgres_data:/var/lib/postgresql/data
#     restart: unless-stopped

#   mongo:
#     image: mongo:7
#     container_name: mongo
#     ports:
#       - "27017:27017"
#     environment:
#       MONGO_INITDB_ROOT_USERNAME: f0DxMZNEQ
#       MONGO_INITDB_ROOT_PASSWORD: JWjB3p7MgOZ4ZCnPxLnEsuPB
#     volumes:
#       - mongo_data:/data/db
#     restart: unless-stopped