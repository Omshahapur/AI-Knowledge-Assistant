from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

import uuid


def get_client():

    return QdrantClient(
        path="qdrant_data"
    )


client = get_client()

COLLECTION_NAME = "documents"


def create_collection():

    if not client.collection_exists(
        COLLECTION_NAME
    ):

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(
                size=768,
                distance=Distance.COSINE
            )
        )

        print(
            "Collection created successfully"
        )

    else:

        print(
            "Collection already exists"
        )


def add_document_chunk(
    text,
    embedding,
    document_name,
    chunk_number,
    total_chunks,
    chat_id
):

    client.upsert(

        collection_name=COLLECTION_NAME,

        points=[

            PointStruct(

                id=str(
                    uuid.uuid4()
                ),

                vector=embedding,

                payload={

                    "text": text,

                    "document_name":
                        document_name,

                    "chunk_number":
                        chunk_number,

                    "total_chunks":
                        total_chunks,

                    "chat_id":
                        chat_id
                }
            )
        ]
    )

    print(
        "Chunk stored successfully"
    )


def search_documents(
    query_embedding,
    chat_id,
    limit=5
):

    chat_filter = Filter(

        must=[

            FieldCondition(

                key="chat_id",

                match=MatchValue(
                    value=chat_id
                )
            )
        ]
    )


    results = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding,

        query_filter=chat_filter,

        limit=limit,

        with_payload=True
    )


    return results.points


def close_client():

    client.close()