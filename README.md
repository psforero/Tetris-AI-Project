# Tetris-AI-Project
AI agent that plays Tetris. Divided into three main parts: the game logic and GUI, the implementation of the feature set and cost function, and the actual agent.

## Initial Setup

This repo uses python3's virtualenvs. See the [documentation](https://docs.python.org/3/tutorial/venv.html).

Create a virtualenv (only do this once)

```
python3 -m venv env
```

## Active Venv

Do this every time before working on this project.

```
source env/bin/activate
```

## Install Dependencies

Do this the first time and when requirements have changed.

```
pip install -m requirements.txt
```

If you you've updated requirements, please add them to the requirements.txt file:

```
pip install whatever
pip freeze > requirements.txt
```

## Run it

```
python GUI.py
```