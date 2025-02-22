import boto3
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
from PIL import Image
from io import BytesIO
import json
import os

# Initialize S3 and FashionCLIP
s3 = boto3.client('s3')
fclip = FashionCLIP('fashion-clip')

def load_images_from_s3(bucket_name, collection_name):
    images = []
    image_keys = []
    
    # List all objects in the collection prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=f"{collection_name}/")
    
    for page in pages:
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            if obj['Key'].endswith('/'):  # Skip directories
                continue
            try:
                response = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                image = Image.open(BytesIO(response['Body'].read()))
                image.verify()  # Verify that it is an image
                images.append(image)
                image_keys.append(obj['Key'])
            except Exception as e:
                print(f"Error loading image {obj['Key']}: {str(e)}")
                continue
    
    return images, image_keys

def handler(event):
    try:
        # Runpod expects input in event['input']
        job_input = event['input']
        
        # Extract collection name and query
        bucket_name = job_input['bucket_name']
        collection_name = job_input['collection_name']
        text_query = job_input['query']
        
        # Load images from S3
        images, image_keys = load_images_from_s3(bucket_name, collection_name)
        
        if not images:
            return {
                "error": f"No images found in collection {collection_name}"
            }
        
        # Create image embeddings
        image_embeddings = fclip.encode_images(images, batch_size=32)
        image_embeddings = np.array(image_embeddings)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        
        # Create text embedding
        text_embedding = fclip.encode_text([text_query], 32)[0]
        
        # Calculate similarity scores
        similarity_scores = text_embedding.dot(image_embeddings.T)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_images = [image_keys[i] for i in sorted_indices]
        
        return {
            "output": {
                "sorted_images": sorted_images,
                "similarity_scores": similarity_scores[sorted_indices].tolist()
            }
        }
        
    except Exception as e:
        return {
            "error": str(e)
        }

if __name__ == "__main__":
    handler({"input": {"query": "test", "bucket_name": "runpod-fclip-ref-collections-storage", "collection_name": "Summer Collection"}})