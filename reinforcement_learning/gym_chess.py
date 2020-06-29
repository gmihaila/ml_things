import gym
import gym_chess
import random

env = gym.make('Chess-v0')
print(env.render())

env.reset()
done = False

while not done:
    action = random.choice(env.legal_moves)
    done = env.step(action)[2]
    display.clear_output(wait=True)
    print(env.render(mode='unicode'))
    

env.close()
