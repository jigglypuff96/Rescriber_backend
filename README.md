
All commands and instructions are for the macOS

# install ollama

download from https://ollama.com 

open terminal 
`ollama run llama3`

`pip install ollama`

make sure there is ollama icon (running ollama app) from the tray menu(the top right of your screen)



`pip install` missing module

`python3 server.py`

modify user_message in client_test.py

`python3 client_test.py`

# Function execution flow
https://drive.google.com/file/d/1uAudLfYgk_KRx3a_f6MpnGL_atTlihEV/view?usp=sharing

# Performance Analysis
## Enable Debug mode:
Quit the running ollama app from the tray menu（top right of your screen）
add `export OLLAMA_DEBUG="1"` in your `~/.zshrc`
(`vim ~/.zshrc`, `export OLLAMA_DEBUG="1"`, `source ~/.zshrc` )
`ollama serve`

# Todo:
refactor code 

calculate FPR & response time,  vs GPT4

change the storage of embeddings.csv (intermediate data)
