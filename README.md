# HOW TO

## Prepare

```sh
sudo apt update
sudo apt install python3-venv ffmpeg -y
```

Create and activate a virtual environment, then install Python dependencies:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```sh
python3 mock_call.py
```