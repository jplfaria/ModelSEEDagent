#!/bin/bash
# ModelSEEDagent Environment Setup Script
#
# Source this file to set up environment variables for optimal CLI experience:
# source examples/setup_environment.sh

echo "🧬 Setting up ModelSEEDagent environment variables..."

# Default LLM Backend - choose from: argo, openai, local
export DEFAULT_LLM_BACKEND="argo"

# Default Model Name - recommended models:
# For Argo: gpt4o, gpt4olatest, gpto1, gpto1mini, gpto3mini, llama-3.1-70b
# For OpenAI: gpt-4o, gpt-4-turbo, gpt-4, gpt-3.5-turbo
export DEFAULT_MODEL_NAME="gpt4o"

# Argo Gateway Configuration
# Replace with your actual ANL username
export ARGO_USER="${USER}"  # Uses your system username, change if needed

# Optional: OpenAI API Key (uncomment and set if using OpenAI)
# export OPENAI_API_KEY="your_openai_api_key_here"

echo "✅ Environment configured:"
echo "   DEFAULT_LLM_BACKEND: $DEFAULT_LLM_BACKEND"
echo "   DEFAULT_MODEL_NAME: $DEFAULT_MODEL_NAME"
echo "   ARGO_USER: $ARGO_USER"

echo ""
echo "🚀 You can now use:"
echo "   modelseed-agent setup --non-interactive  # Uses defaults"
echo "   modelseed-agent switch argo             # Quick switch to Argo"
echo "   modelseed-agent switch argo --model gpto1  # Switch to reasoning model"
echo ""
echo "💡 Pro tip: Add these exports to your ~/.bashrc or ~/.zshrc for permanent setup"
