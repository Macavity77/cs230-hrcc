
course project for cs230 deep learning at Stanford

CS230-HandWrittenChineseCharacterReM1 -- the basic 4-layer DL model for milestone 1, running locally on my mac, took about 3h. Since the number of classes is over 3000, it will be really slow for this simply model

Everything else was build after milestone 1.

For milestone 2, the basic model was based on the model from this repo: https://github.com/brucegarro/chinese-character-recognition.git. And modified

In AWS EC2 instance, please choose tensorflow_p36 env.
The training process last about 1h 30min(without checkpointers). Simply run `python train.py`
For testing, simply run `python test.py`
Test data is not inclued in this repo since it's too large, it's from CASIA offline database HWDB 1.1 
