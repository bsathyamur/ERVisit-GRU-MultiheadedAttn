# Emergency room visit prediction model from claims data

#### Objective:

GRU Multiheaded attention model for predicting patient hospital emergency visit based on claims data. 

### Model execution steps

Please follow the below steps for the execution of the model

1. Create the following folder structure below executing the code
   -config\
   -inputFiles\ccs\
   -inputFiles\claims\
   -inputFiles\enroll\
   -output\
   -plot\

2. Place the following input files in the below mentioned folders:
   -default.yml in config\
   -claims_xxxx.dat data files in inputFiles\claims\
   -enroll_synth.dat data file in inputFiles\enroll\
   -ccs_xwlk.csv and ccs_desc.csv in inputFiles\ccs\
   
3. First run the datapreprocess.py file from command prompt [python datapreprocess.py]

5. Next run the models.py file from command prompt [python models.py]

#### Model Design

![img1](https://github.com/bsathyamur/ERVisit-GRU-MultiheadedAttn/blob/main/model%20diagram.png)
