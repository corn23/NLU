### Brief Intro
Currently, all functions are in ```NLM.py```, where you can change some basic parameters. and tensorflow model is defined in ```RNN.py```.

 Type ```python NLM.py``` and it will run. For the full train set (200,000), it will cost 2.5 hour per epoch if ```is_add_layer=False```, 3.2 hour if ```is_add_layer=True``` on **leonhard cluster**. 

### Future work

Restructure the code. Create a class for the RNN with three methods, **train**, **evaluate** and **generate**.


