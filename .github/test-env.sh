#!/bin/bash
# CI test environment setup for YMemo
# This script sets up a complete test environment with mock services and credentials

set -e  # Exit on any error

echo "🧪 Setting up YMemo CI test environment..."

# Test environment indicators
export SKIP_AWS_VALIDATION=true
export MOCK_SERVICES=true
export TESTING=true
export CI=true
export PYTEST_RUNNING=true

# Logging configuration (reduce noise in CI)
export LOG_LEVEL=WARNING

# Mock AWS credentials (required for boto3 initialization)
export AWS_ACCESS_KEY_ID=test-access-key-id
export AWS_SECRET_ACCESS_KEY=test-secret-access-key
export AWS_DEFAULT_REGION=us-east-1
export AWS_REGION=us-east-1

# YMemo-specific test configuration  
export TRANSCRIPTION_PROVIDER=aws
export CAPTURE_PROVIDER=file
export AUDIO_SAMPLE_RATE=16000
export AUDIO_CHANNELS=1

# Additional AWS environment variables for comprehensive mocking
export AWS_SESSION_TOKEN=test-session-token
export AWS_SECURITY_TOKEN=test-security-token

# Create fake AWS credentials directory structure
echo "📁 Creating mock AWS credentials directory..."
mkdir -p ~/.aws

# Create AWS credentials file
cat > ~/.aws/credentials << EOF
[default]
aws_access_key_id = test-access-key-id
aws_secret_access_key = test-secret-access-key
region = us-east-1

[test]
aws_access_key_id = test-access-key-id
aws_secret_access_key = test-secret-access-key
region = us-east-1
EOF

# Create AWS config file
cat > ~/.aws/config << EOF
[default]
region = us-east-1
output = json

[profile test]
region = us-east-1
output = json
EOF

# Set permissions (AWS CLI expects specific permissions)
chmod 600 ~/.aws/credentials
chmod 600 ~/.aws/config

# Additional environment variables for boto3
export AWS_SHARED_CREDENTIALS_FILE=~/.aws/credentials
export AWS_CONFIG_FILE=~/.aws/config

echo "✅ YMemo CI test environment configured successfully!"
echo "🔧 Environment summary:"
echo "   - AWS validation: DISABLED"
echo "   - Mock services: ENABLED" 
echo "   - Log level: WARNING"
echo "   - Provider: $TRANSCRIPTION_PROVIDER"
echo "   - Audio: ${AUDIO_SAMPLE_RATE}Hz, ${AUDIO_CHANNELS} channel(s)"
echo ""