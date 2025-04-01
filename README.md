# RL

Implementations of PPO, IPPO, and DAgger from Pytorch and Numpy. Most methods are based on the Stable Baselines 3 library.        

Now moved to Gymnasium!             
Two main files, one for Gymnasium environments, one for my custom environments. Environments in the repo but not in the main files are not guaranteed to work with the new setup. One set of hyperparameters is used for all environments, so while it's not optimized for each environment, it's good enough to work. All the old demo files have been moved to a separate folder and most likely don't work anymore.        
             
To Do: DAgger, PPO finetune, and IPPO+LSTM are not working right now, however I'm ignoring that to work on DreamerV3.      
            
Implementations of PPO, IPPO, and DAgger from Pytorch and Numpy. Most methods are based on the Stable Baselines 3 library.     
           
           
Trained PPO agents (deleted some of the clips whoops):           

https://github.com/user-attachments/assets/3758b081-8522-4acb-8ec8-cdfcba9e77c5

https://github.com/user-attachments/assets/be57bc7e-f588-4a62-9884-cd277883756b



Note:      
Main code *has* to be put into an "if \_\_name__=='\_\_main__'" statement or else multiprocessing will throw a fit.

RL Explanations: 
https://docs.google.com/document/d/1sIpagn56Lj0PETUN_K0YCWtSnPrsWdvH9FDiSf9ZSZ8/edit
