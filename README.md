# llama

# various ops on llama llm

# linux install

curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
ollama pull llama3.1
ollama pull mistral
python venv -v venv # virtual environment
pip -r requirements.txt
python llamaemb_cntx.py

# to test prompt tuning for CLIP

python multi_dream.py
