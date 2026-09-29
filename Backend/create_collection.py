from qdrant_service import client, create_collection


create_collection()

client.close()

print("Qdrant closed successfully")