# deactivate any currently active virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# install uv if not already installed
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# create a new virtual environment in the 'venv' directory
uv venv
source .venv/bin/activate

# install requirements
uv pip install -r requirements.txt