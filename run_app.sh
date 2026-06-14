#!/bin/bash
# run_app.sh — Launcher for DeepGuard AI streamlit interface using the correct deepguard environment.

echo "=========================================================="
echo "🛡️  DeepGuard AI Streamlit App Launcher"
echo "=========================================================="

# Resolve conda path
if ! command -v conda &> /dev/null; then
    echo "⚠️  Conda command not found. Trying to find default installation paths..."
    # Try sourcing conda from common installation directories
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/usr/local/anaconda3/etc/profile.d/conda.sh"
    fi
fi

# Check if conda command is now available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in your PATH."
    echo "Please activate conda manually in your shell and run 'streamlit run app.py'."
    exit 1
fi

# Check if conda env deepguard exists
if conda env list | grep -q "deepguard"; then
    echo "✨ Activating 'deepguard' conda environment..."
    # We must use "conda activate" or source the activation script
    # Sourcing profile.d/conda.sh makes activate command work in non-interactive scripts
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        source "/opt/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate deepguard
    
    # Run streamlit from the active environment
    ST_PATH=$(which streamlit)
    if [ -n "$ST_PATH" ] && [[ "$ST_PATH" == *"deepguard"* ]]; then
        echo "🚀 Starting Streamlit app..."
        streamlit run app.py
    else
        # If active env python does not have streamlit, try running the specific bin
        if [ -f "/opt/anaconda3/envs/deepguard/bin/streamlit" ]; then
            echo "🚀 Starting Streamlit app via absolute path..."
            /opt/anaconda3/envs/deepguard/bin/streamlit run app.py
        else
            echo "⚠️  Streamlit not found in 'deepguard' environment. Installing it now..."
            /opt/anaconda3/envs/deepguard/bin/pip install streamlit
            echo "🚀 Starting Streamlit app..."
            /opt/anaconda3/envs/deepguard/bin/streamlit run app.py
        fi
    fi
else
    echo "❌ Conda environment 'deepguard' was not found."
    echo "Please create the 'deepguard' environment using:"
    echo "  conda create -n deepguard python=3.10"
    echo "  conda activate deepguard"
    echo "  pip install -r requirements_mac.txt"
    echo "Then run this launcher again."
    exit 1
fi
