#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi


export GOOGLE_API_KEY=${GOOGLE_API_KEY}
export GOOGLE_MODEL=${GOOGLE_MODEL}

export OPENAI_API_KEY=${OPENAI_API_KEY}
export OPENAI_MODEL=${OPENAI_MODEL}

export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
export ANTHROPIC_MODEL=${ANTHROPIC_MODEL}

export GROQ_API_KEY=${GROQ_API_KEY}
export GROQ_MODEL=${GROQ_MODEL}

export LAMBDA_LAB_API_KEY=${LAMBDA_LAB_API_KEY}
export LAMBDA_LAB_MODEL=${LAMBDA_LAB_MODEL}

# Check if a specific test function is provided
if [ -n "$1" ]; then
    go test -v -run $1 ./...
else
    echo "No test function specified. Running ALL tests."
    echo "To run specific tests: ./test.sh TestFunctionName"
    echo -n "Continue? [y/N] "
    read answer
    
    if [ "$answer" != "${answer#[Yy]}" ]; then
        go test -v ./...
    else
        echo "Test execution cancelled"
        exit 0
    fi
fi
