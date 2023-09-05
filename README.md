# DVC Question-Answering Chat Bot

This chat bot is built on top of LangChain and uses the DVC documentation and Discord discussions as data.

This is chatbot about DVC where the training pipeline was built using DVC.

# Environment Setup

First you need to do a git pull of the code:
```shell
git clone git@github.com:iterative/llm-demo.git
cd llm-demo
```

The training run is all logged in DVC on an S3 store. So, if you are already authenticated on AWS (check with `aws sts get-caller-identity`) you can get all the data with:
```shell
dvc pull
```

In order to set your environment up to run the code here, first install all requirements in a virtual env:
```shell
virtualenv env --python=python3.9
source env/bin/activate
pip install -r requirements.txt
```

Then set your OpenAI API key (if you don't have one, get one [here](https://beta.openai.com/playground)):
```shell
  export OPENAI_API_KEY=....
```
The preceeding spaces prevent the API key from staying in your bash history if that is [configured](https://stackoverflow.com/questions/6475524/how-do-i-prevent-commands-from-showing-up-in-bash-history).

# Running

Now you should be ready to re-run the training pipeline. Assuming you have not changed anything, nothing should need to run. Everything can be re-used for the DVC pull:
```shell
dvc repro
```

Now you can startup the web UI using:
```shell
streamlit run main.py
```
The log of interactions can be found in `chat.log`.
