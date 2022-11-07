import time
import random
import gym

env = gym.make("gym_nav:basic-nav-v0",
               args={"host": "151.97.13.152", "port": 3000
                     })

levels = ["ForestLevel", "FlatLevel"]

env.reset(args={"Difficuty": 0.1, "LevelName": "ForestLevel"})

'''
states = []
state, reward, done, _ = env.step(0)
states += [state]
time.sleep(0.5)
#state["rgb_img"].show()

states = []
state, reward, done, _ = env.step(1)
states += [state]
#state.show()
time.sleep(2)

state, reward, done, _ = env.step(3)
states += [state]
#state.show()
time.sleep(1)

state, reward, done, _ = env.step(0)
states += [state]
#state.show()

state["rgb_img"].show()

print("hello")
'''

for i in range(10):
    env.reset(args={"Difficuty": 0.1, "LevelName": levels[1]})

    for i in range(20):
        action = random.randrange(0, env.action_space.n)
        state, reward, done, _ = env.step(action)
        #d_angle = (state["angle"] * 180) / 3.14
        #if d_angle > 5:
        time.sleep(random.random() * 3)

# TODO: Fix problm with Image reception.
