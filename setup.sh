set -e

# Check the API key has been set
[ -n "${OPENAI_API_KEY+1}" ]

# Make sure any install is inside a virtual env
export PIP_REQUIRE_VIRTUALENV=true

# Ensure requirements are up to date
pip install -r requirements.txt
