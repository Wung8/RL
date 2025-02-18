# RL

Currently under development, added PPO+LSTM, trying to find more robust hyperparameters for PPO+CNN+LSTM
DAgger, PPO finetune, and IPPO+LSTM are most likely not working. 

Implementations of PPO, IPPO, and DAgger from Pytorch and Numpy. Most methods are based on the Stable Baselines 3 library.     
All demo files should work out of the box.
           
           
Trained PPO agents:         

![Untitled video - Made with Clipchamp (1)](https://github.com/user-attachments/assets/ef0a4d8b-bafd-406a-9028-88e794814334)
           
![Untitled video - Made with Clipchamp (2)](https://github.com/user-attachments/assets/87ed8a0c-32aa-47b0-8200-cdd14322831e)

![TMaze (2)](https://github.com/user-attachments/assets/70b5ddd1-9411-4745-a64f-4fc37b7ee304)

![platformer](https://github.com/user-attachments/assets/1e846568-a62c-4a3e-b7bd-c73c5c867535)


Note:      
Main code *has* to be put into an "if \_\_name__=='\_\_main__'" statement or else multiprocessing will throw a fit.

RL Explanations: 
https://docs.google.com/document/d/1sIpagn56Lj0PETUN_K0YCWtSnPrsWdvH9FDiSf9ZSZ8/edit
