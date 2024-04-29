echo [$(date)]: "START"

echo [$(date)]: "Creating environment with Python 3.12 version"

python -m pip install virtualenv
python -m venv myenv 

echo [$(date)]: "Activating environment"

# Unix-like systems (Linux, macOS)
# source myenv/bin/activate

# Activating virtual environment on Windows with the full path
source ./myenv/Scripts/activate

echo [$(date)]: "Installing the dev requirements"

pip install -r requirements_dev.txt

echo [$(date)]: "END"