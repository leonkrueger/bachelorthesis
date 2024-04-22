# python3.9 -m venv venv
# python3.9 -m pip install --upgrade setuptools pip distlib
# python3.9 -m pip install -r finetuning.requirements.txt

source venv/bin/activate
python3.9 fine_tuning.py
deactivate