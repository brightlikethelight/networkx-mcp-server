#!/bin/bash
# Test Redis persistence

echo "=== Testing Redis Storage Persistence ==="
echo

# Start Redis
echo "1. Starting Redis container..."
docker-compose -f docker-compose.test.yml up -d redis

# Wait for Redis to be ready
echo "2. Waiting for Redis to be ready..."
sleep 5

# Test Redis connection
echo "3. Testing Redis connection..."
docker-compose -f docker-compose.test.yml exec redis redis-cli ping

# Run storage test with Redis
echo
echo "4. Creating test graphs with Redis backend..."
REDIS_URL=redis://localhost:6379 python test_storage_persistence.py

echo
echo "5. Verifying persistence after restart..."
REDIS_URL=redis://localhost:6379 python test_storage_persistence.py --verify

echo
echo "6. Cleaning up..."
read -p "Press Enter to stop Redis and clean up..."
docker-compose -f docker-compose.test.yml down

echo "✅ Test complete!"