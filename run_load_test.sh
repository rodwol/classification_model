#!/bin/bash

# run_load_test.sh - Run load tests with different container counts

set -e

echo " Starting Load Testing Experiment"

# Test different container counts
CONTAINER_COUNTS=(1 2 3 4 5)

for containers in "${CONTAINER_COUNTS[@]}"; do
    echo ""
    echo " Testing with $containers containers..."
    
    # Deploy with specific container count
    ./deploy.sh $containers
    
    # Wait for services to be healthy
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Run locust test
    echo " Starting load test..."
    docker-compose exec locust locust -f locustfile.py --host http://nginx \
        --users 100 --spawn-rate 10 --run-time 5m --headless \
        --csv=results_${containers}_containers
    
    # Collect metrics
    echo " Collecting metrics..."
    curl -s http://localhost:8000/metrics > metrics_${containers}.json
    
    echo " Completed test with $containers containers"
    sleep 10
done

echo ""
echo " All load tests completed!"
echo " Results saved to results_*_containers.csv"