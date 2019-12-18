# AlphaZero_Gomoku-tensorflow

### Create virtual environment
```
virtualenv --system-site-packages -p python3 /Users/will/Projects/venv
```

### Activate virttual enviroonment
```
source /Users/will/Projects/venv/bin/activate

pip install --upgrade tensorflow==1.15
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c 'import tensorflow as tf; print(tf.__version__)'
```

Forked from [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) with some changes:  

* rewrited the network code with tensorflow
* trained with 11 * 11 board
* added a GUI

## Usage
To play with the AI

	$ python human_play.py
	
To train the model:

	$ python train.py


### Example of Game

![Example](https://github.com/zouyih/AlphaZero_Gomoku-tensorflow/blob/master/example.gif)  

there's another interesting implementation of reinforcement learning [DQN-tensorflow-gluttonous_snake](https://github.com/zouyih/DQN-tensorflow-gluttonous_snake)