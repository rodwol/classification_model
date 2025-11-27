#!/bin/bash

# deploy.sh - Cloud deployment script

set -e

echo " Deploying Respiratory Sound Classification System"

# Configuration
CLUSTER_NAME="respiratory-cluster"
SERVICE_NAME="respiratory-api"
DOCKER_IMAGE="your-registry/respiratory-sound-api:latest"
CONTAINER_COUNT=${1:-1}

echo "Deploying $CONTAINER_COUNT containers..."

# Build and push image (for cloud deployment)
echo "Building Docker image..."
docker build -t $DOCKER_IMAGE .

# For AWS ECS (example)
# aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
# docker push $DOCKER_IMAGE

# Update service with desired container count
# aws ecs update-service --cluster $CLUSTER_NAME --service $SERVICE_NAME --desired-count $CONTAINER_COUNT

# For local Docker Compose deployment
echo "Starting $CONTAINER_COUNT containers with Docker Compose..."
docker-compose up -d --scale respiratory-api=$CONTAINER_COUNT

echo " Deployment completed with $CONTAINER_COUNT containers"
echo " API URL: http://localhost:8000"
echo " Load testing UI: http://localhost:8089"