try:
    import Image
except ImportError:
    from PIL import Image
import gym, PIL
from IPython import display

env = gym.make('SpaceInvaders-v0')
array = env.reset()
frame = PIL.Image.fromarray(env.render(mode='rgb_array'))
display.display(frame)

for _ in range(1000):
    action = env.action_space.sample()
    frame = PIL.Image.fromarray(env.render(mode='rgb_array'))
    display.display(frame)
    display.clear_output(wait=True)
    env.step(action)
