from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import numpy as np
import io
import tempfile
import os
from typing import List
import uvicorn

app = FastAPI(
    title="Voice Embedding API",
    description="Convert audio files to vector embeddings using Wav2Vec2",
    version="1.0.0"
)

# Global variables to store model (loaded once at startup)
processor = None
model = None

@app.on_event("startup")
async def load_model():
    """Load the Wav2Vec2 model at startup"""
    global processor, model
    print("Loading Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    print("✅ Model loaded successfully!")

@app.get("/")
async def root():
    return {"message": "Voice Embedding API", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/embed")
async def create_embedding(file: UploadFile = File(...)):
    """
    Convert an audio file to a 768-dimensional embedding vector
    
    Accepts: MP3, WAV, FLAC, M4A and other common audio formats
    Returns: JSON with embedding vector and metadata
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file type
    allowed_types = ['audio/', 'application/octet-stream']
    if not any(file.content_type.startswith(t) for t in allowed_types):
        if not file.filename.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg')):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload an audio file."
            )
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Save to temporary file (librosa needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        try:
            # Load and process audio
            audio, sr = librosa.load(tmp_path, sr=16000)
            
            # Check if audio is valid
            if len(audio) == 0:
                raise HTTPException(status_code=400, detail="Invalid or empty audio file")
            
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            
            # Get embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            # Convert to list for JSON serialization
            embedding_list = embedding.numpy().tolist()
            
            return {
                "filename": file.filename,
                "embedding_size": len(embedding_list),
                "audio_duration": len(audio) / 16000,  # Duration in seconds
                "sample_rate": sr,
                "embedding": embedding_list
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/embed-batch")
async def create_embeddings_batch(files: List[UploadFile] = File(...)):
    """
    Convert multiple audio files to embeddings in a single request
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files per batch.")
    
    results = []
    
    for file in files:
        try:
            # Read the uploaded file
            contents = await file.read()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                tmp_file.write(contents)
                tmp_path = tmp_file.name
            
            try:
                # Load and process audio
                audio, sr = librosa.load(tmp_path, sr=16000)
                
                if len(audio) == 0:
                    results.append({
                        "filename": file.filename,
                        "error": "Invalid or empty audio file"
                    })
                    continue
                
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                
                embedding_list = embedding.numpy().tolist()
                
                results.append({
                    "filename": file.filename,
                    "embedding_size": len(embedding_list),
                    "audio_duration": len(audio) / 16000,
                    "sample_rate": sr,
                    "embedding": embedding_list
                })
                
            finally:
                os.unlink(tmp_path)
                
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

@app.post("/similarity")
async def calculate_similarity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    Calculate cosine similarity between two audio files
    """
    try:
        # Process first file
        contents1 = await file1.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file1:
            tmp_file1.write(contents1)
            tmp_path1 = tmp_file1.name
        
        # Process second file
        contents2 = await file2.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file2:
            tmp_file2.write(contents2)
            tmp_path2 = tmp_file2.name
        
        try:
            # Get embeddings for both files
            embeddings = []
            filenames = [file1.filename, file2.filename]
            paths = [tmp_path1, tmp_path2]
            
            for path in paths:
                audio, sr = librosa.load(path, sr=16000)
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                    embeddings.append(embedding.numpy())
            
            # Calculate cosine similarity
            embedding1 = embeddings[0]
            embedding2 = embeddings[1]
            
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            return {
                "file1": file1.filename,
                "file2": file2.filename,
                "cosine_similarity": float(similarity),
                "similarity_percentage": float(similarity * 100)
            }
            
        finally:
            os.unlink(tmp_path1)
            os.unlink(tmp_path2)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating similarity: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)