# Tetris-AI-Project
AI agent that plays Tetris. Divided into three main parts: the game logic and GUI, the implementation of the feature set and cost function, and the actual agent.

## Initial Setup

This repo uses python 3's virtualenvs. See the [documentation](https://docs.python.org/3/tutorial/venv.html).

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
pip install -r requirements.txt
```

If you you've updated requirements, please add them to the requirements.txt file:

```
pip install whatever
pip freeze > requirements.txt
```

## Running it

To play as a human:

```
python GUI.py
```

To see an agent play:

```
python GUI.py --agent <agentname>

# Agents available
'random': Random agent
'human' : Human agent
'hand'  : Hand-tuned agent
'neural': Neural network agent
```

To train one of the AI agents:

```
python GUI.py --agent <agentname> --outfile <filename>
```

Use `--trials` to change the number of training iterations

```
python GUI.py --agent <agentname> --outfile <filename> --trials 5
```

Use `--games` to change the number of games played

```
python GUI.py --agent <agentname> --games 50
```

To run as a pretrained agent:

```
python GUI.py --agent <agentname> --infile <filename>
```