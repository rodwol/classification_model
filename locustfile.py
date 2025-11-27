# locustfile.py
from locust import HttpUser, task, between, TaskSet
import random
import json
import time

class RespiratorySoundUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Check health on start"""
        self.client.get("/health")
    
    @task(3)
    def health_check(self):
        """Health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    def single_prediction(self):
        """Single prediction request"""
        # Simulate different audio file sizes
        file_sizes = [1024, 2048, 4096, 8192]  # bytes
        file_size = random.choice(file_sizes)
        
        # Generate random audio-like data
        audio_data = bytearray(random.getrandbits(8) for _ in range(file_size))
        
        files = {'file': ('test_audio.wav', audio_data, 'audio/wav')}
        
        with self.client.post("/predict", 
                             files=files, 
                             name="Single Prediction",
                             catch_response=True) as response:
            
            if response.status_code == 200:
                # Check response time
                if response.elapsed.total_seconds() > 10:
                    response.failure("Response too slow")
                else:
                    response.success()
                    
                # Log prediction details
                try:
                    result = response.json()
                    self.environment.events.request.fire(
                        request_type="POST",
                        name="Prediction Success",
                        response_time=response.elapsed.total_seconds() * 1000,
                        response_length=len(response.content),
                        exception=None,
                        context={**result}
                    )
                except:
                    pass
            else:
                response.failure(f"Prediction failed: {response.status_code}")
    
    @task(1)
    def batch_prediction(self):
        """Batch prediction request"""
        files = []
        num_files = random.randint(2, 5)
        
        for i in range(num_files):
            file_size = random.choice([1024, 2048])
            audio_data = bytearray(random.getrandbits(8) for _ in range(file_size))
            files.append(('files', (f'test_audio_{i}.wav', audio_data, 'audio/wav')))
        
        with self.client.post("/batch-predict", 
                             files=files,
                             name="Batch Prediction",
                             catch_response=True) as response:
            
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")
    
    @task(2)
    def model_info(self):
        """Get model information"""
        self.client.get("/model-info")

class HighLoadUser(HttpUser):
    wait_time = between(0.1, 0.5)
    
    @task(10)
    def rapid_predictions(self):
        """Rapid fire predictions for stress testing"""
        file_size = random.choice([512, 1024])
        audio_data = bytearray(random.getrandbits(8) for _ in range(file_size))
        files = {'file': ('stress_test.wav', audio_data, 'audio/wav')}
        
        self.client.post("/predict", files=files, name="Stress Test Prediction")