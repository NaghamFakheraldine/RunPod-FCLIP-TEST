import runpod
import boto3
from fashion_clip.fashion_clip import FashionCLIP
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image
import os
import time
from functools import lru_cache

# Initialize S3 client
s3_client = boto3.client('s3',
    region_name=os.environ['AWS_DEFAULT_REGION'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY']
)

@lru_cache(maxsize=1000)
def get_image_embedding(image_path):
    return fclip.encode_images([image_path], batch_size=1)[0]

def download_and_process_image(image_key, bucket):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=image_key)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))
        # Resize image to smaller size for faster processing
        image.thumbnail((224, 224))  # Standard CLIP input size
        return image_key, image
    except Exception as e:
        print(f"Error processing image {image_key}: {str(e)}")
        return None

def load_images_from_s3(bucket, prefix):
    print(f"Loading images from S3 bucket: {bucket}, prefix: {prefix}")
    
    # List all objects in the S3 bucket with given prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    image_keys = []
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_keys.append(obj['Key'])
    
    print(f"Found {len(image_keys)} images")
    
    # Download and process images in parallel
    images = []
    with ThreadPoolExecutor(max_workers=20) as executor:  # Increased from 10
        futures = []
        # Process in batches of 50
        batch_size = 50
        for i in range(0, len(image_keys), batch_size):
            batch = image_keys[i:i + batch_size]
            futures.extend([executor.submit(download_and_process_image, key, bucket) 
                          for key in batch])
        for future in futures:
            result = future.result()
            if result:
                images.append(result)
    
    return images, image_keys

def handler(event):
    try:
        start_time = time.time()
        print("Handler started")
        
        # Initialize model at first request
        global fclip
        if not 'fclip' in globals():
            print("Initializing FashionCLIP model")
            fclip = FashionCLIP('fashion-clip')
        
        # Extract parameters from event
        bucket = event['input']['bucket']
        prefix = event['input']['prefix']
        query = event['input']['query']
        
        print(f"Processing request - Bucket: {bucket}, Prefix: {prefix}, Query: {query}")
        
        # Load and process images
        images, image_keys = load_images_from_s3(bucket, prefix)
        print(f"Loaded {len(images)} images in {time.time() - start_time:.2f} seconds")
        
        # Create image embeddings
        image_paths = [img[1] for img in images]
        batch_size = 64  # Increased from 32
        image_embeddings = fclip.encode_images(image_paths, batch_size=batch_size)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        print(f"Created embeddings in {time.time() - start_time:.2f} seconds")
        
        # Create text embedding
        text_embedding = fclip.encode_text([query], 32)[0]
        print("Text embedded")
        
        # Calculate similarity scores
        similarity_scores = text_embedding.dot(image_embeddings.T)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        # Map indices to S3 keys
        results = [image_keys[idx] for idx in sorted_indices]
        print("Top results:", results[:10])  # Print first 10 results
        print(f"Search completed in {time.time() - start_time:.2f} seconds")
        
        return {
            "status": "success",
            "results": results,
            "metrics": {
                "total_time": time.time() - start_time,
                "total_images": len(images)
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"status": "error", "error": str(e)}

runpod.serverless.start({"handler": handler})
