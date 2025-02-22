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
from botocore.exceptions import ClientError
from time import sleep

# Initialize S3 client
s3_client = boto3.client('s3',
    region_name=os.environ['AWS_DEFAULT_REGION'],
    aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    config=boto3.Config(
        max_pool_connections=50,  # Increase from default 10
        connect_timeout=60,
        read_timeout=60,
        retries={'max_attempts': 3}
    )
)

@lru_cache(maxsize=1000)
def get_image_embedding(image_path):
    return fclip.encode_images([image_path], batch_size=1)[0]

def download_and_process_image(image_key, bucket, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = s3_client.get_object(Bucket=bucket, Key=image_key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail((224, 224))
            return image_key, image
        except ClientError as e:
            if attempt == max_retries - 1:
                print(f"Error processing image {image_key} after {max_retries} attempts: {str(e)}")
                return None
            sleep(2 ** attempt)  # Exponential backoff
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
    with ThreadPoolExecutor(max_workers=min(50, len(image_keys))) as executor:
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
        
        # Extract parameters from event
        bucket = event['input']['bucket']
        user_id = event['input']['user_id']
        prefix = event['input']['prefix']
        query = event['input']['query']
        
        # Combine user_id and prefix for the full S3 path
        full_prefix = f"{user_id}/{prefix}"
        
        print(f"Processing request - Bucket: {bucket}, User: {user_id}, Prefix: {prefix}, Query: {query}")
        
        # Start loading images in a separate thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            images_future = executor.submit(load_images_from_s3, bucket, full_prefix)
            
            # While images are loading, initialize the model
            global fclip
            if not 'fclip' in globals():
                print("Initializing FashionCLIP model")
                fclip = FashionCLIP('fashion-clip')
            
            # Wait for images to complete loading
            images, image_keys = images_future.result()
            print(f"Loaded {len(images)} images in {time.time() - start_time:.2f} seconds")
        
        image_paths = [img[1] for img in images]
        batch_size = 64
        image_embeddings = fclip.encode_images(image_paths, batch_size=batch_size)
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, ord=2, axis=-1, keepdims=True)
        print(f"Created embeddings in {time.time() - start_time:.2f} seconds")
        
        text_embedding = fclip.encode_text([query], 32)[0]
        print("Text embedded")
        
        similarity_scores = text_embedding.dot(image_embeddings.T)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        print(f"Search completed in {time.time() - start_time:.2f} seconds")
        
        return {
            "sorted_indices": sorted_indices.tolist(),
            "image_keys": image_keys,
            "metrics": {
                "total_time": time.time() - start_time,
                "total_images": len(images)
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
