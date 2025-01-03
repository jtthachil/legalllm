from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

try:
    client = QdrantClient(
        url="https://34539fef-443a-4399-a418-9a5a44bdf3c6.us-west-1-0.aws.cloud.qdrant.io:6333", 
        api_key="EXWLgeyMKAkkdMoiIyNRipxhE0iiqWnx032linLWDckYqrAKiwwh8Q",
    )

    # Test connection by getting collection info
    print("Testing connection...")
    collection_info = client.get_collections()
    print(f"Successfully connected to Qdrant! Found {len(collection_info.collections)} collections")

    # Test operations
    print("\nListing collections:")
    for collection in collection_info.collections:
        print(f"- {collection.name}")

    print("\nTrying to create a collection...")
    try:
        client.create_collection(
            collection_name="legal_knowledge",
            vectors_config={
                "size": 1536,
                "distance": "Cosine"
            }
        )
        print("Collection 'legal_knowledge' created successfully!")
    except UnexpectedResponse as e:
        if "already exists" in str(e):
            print("Collection 'legal_knowledge' already exists")
        else:
            raise e

except Exception as e:
    print(f"Error connecting to Qdrant: {str(e)}")