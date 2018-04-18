### Brief Intro
#### project structure
- **NLM.py** main script. 
- **RNN.py** define the LSTM graph
- **model.py** define three scenario: **train -t**, **evaluate -e** and **generate -g**
- **get_cfg.py** define all the configuration parameters.
- **utils.py** data load functions

#### quick-run command 
 must specify which scenario you want to implement (-t, -e or -g). Eg. training mode:
 
>` python NLM.py -t 1`
 
 or evaluate mode or generate mode
 
> `python NLM.py -e 1 -sess_path run/1523866215`
> `python NLM.py -g 1 -sess_path run/1523866215`

 In these two modes, ```-sess_path``` parameter must be provided so that the program know which tensorflow model you want to use. ```run/1523866215``` will be generated automatically in **training** mode.
 
#### more details
There are more options you could choose for running.
type `python NLM.py` for the help function.

#### running time 
On Leonhard cluster, if set `is_add_layer=False`, it will cost around **2.5 hour** per epoch (**200,000** training texts). If new layer is added, that number would be **3.5 hour**.

Do a quick calculation before starting your jobs in cluster:).

### Future work

Restructure the code. Create a class for the RNN with three methods, **train**, **evaluate** and **generate**.


