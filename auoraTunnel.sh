#!/bin/bash

# Aurora DB Tunneling Script (Prod Only)
#
# USAGE:
#   sh ./auoraTunnel.sh
#
# REQUIREMENTS:
# - AWS CLI
# - AWS Session Manager Plugin
# - aws-sso-util

# Constants
REGION="eu-central-1"
SSM_ID="i-0c3ca776bd5e0fbed"
HOST_NAME="cdk-hg-insertion-api-prod-au-auroracluster23d869c0-9axwp7gm4sxz.cluster-ro-cgjwhtkgzbvi.eu-central-1.rds.amazonaws.com"
AWS_PROFILE="aws_hg-insertion-funnel-prod.ssm-readonly-user"
PORT_NUMBER="5432"

# Function to check if a profile exists
check_and_create_profile() {
  local profile_name=$1
  aws configure list-profiles | grep -q "^${profile_name}$"
  if [ $? -ne 0 ]; then
    echo "🔍 Profile ${profile_name} does not exist. Creating profile... ✨"
    aws configure --profile ${profile_name}
  fi
}

# Function to log in to AWS SSO and export credentials
setup_aws_credentials() {
  if [ -n "$AWS_PROFILE" ]; then
    aws-sso-util login
    echo "🔐 Exporting AWS credentials for profile $AWS_PROFILE..."
    aws configure export-credentials --profile $AWS_PROFILE > /dev/null
    if [ $? -ne 0 ]; then
      echo "❌ Failed to export credentials for profile $AWS_PROFILE."
      exit 1
    fi
  fi
}

# Function to start the AWS SSM Session
start_tunnel() {
  aws ssm start-session --target $SSM_ID \
    --region $REGION \
    --profile $AWS_PROFILE \
    --document-name AWS-StartPortForwardingSessionToRemoteHost \
    --parameters "{
      \"portNumber\": [\"5432\"],
      \"localPortNumber\": [\"$PORT_NUMBER\"],
      \"host\": [\"$HOST_NAME\"]
    }"
}

# Main Execution
check_and_create_profile $AWS_PROFILE
setup_aws_credentials

echo "🚀 Starting tunnel to Aurora PROD DB instance..."
echo "🔑 Using SSM ID: $SSM_ID"
echo "🔑 Using Host Name: $HOST_NAME"
echo "🔑 Using AWS Profile: $AWS_PROFILE"

start_tunnel

if [ $? -ne 0 ]; then
  echo "❌ Failed to start tunnel."
  exit 1
fi

echo "✅ Tunnel closed."
