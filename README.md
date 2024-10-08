# RL

Implementations of REINFORCE, Actor Critic, and Proximal Policy Optimization from Pytorch and Numpy. Most methods are based off of the Stable Baselines 3 library.     
All demo files should work out of the box.
           
           
Trained PPO agent playing flappy bird and single player slime volleyball:     

![Untitled video - Made with Clipchamp (1)](https://github.com/user-attachments/assets/ef0a4d8b-bafd-406a-9028-88e794814334)
           
![Untitled video - Made with Clipchamp](https://github.com/user-attachments/assets/4b3f1469-c287-4c1a-9070-da5fb07fbaf9)           
           

Parameters of built in envs:           
cartpole: observation_space=4, action_space=3           
flappy_bird: observation_space=5, action_space=2           
flappy_bird_img: observation_space=(40,30,40), action_space=2   (will have to make a custom model, default model doesn't work with images)                 
slime_volleyball: WIP           
slime_volleyball_single_player: observation_space=6, action_space=4  

Note:      
Main code *has* to be put into an "if \_\_name__=='\_\_main__'" statement or else multiprocessing will throw a fit.

RL Explanations: 
https://docs.google.com/document/d/1sIpagn56Lj0PETUN_K0YCWtSnPrsWdvH9FDiSf9ZSZ8/edit

To Do:           
Update REINFORCE and Actor Critic to work with updated vec_env_handler         
Implement PPO with self play
