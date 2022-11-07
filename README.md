# Reinforcement Learning in Unstructured Environments
## Reinforcement Learning algorithms in the MIDGARD simulator

Autonomous navigation is a very important area in the huge domain of mobile autonomous vehicles. In general, the autonomous navigation is a combination of complex systems that assists in carrying out decision-making based on surrounding situations, and accordingly operate the vehicle without human interventions. Many vehicles such as drones, robots, cars and boats have been installed with such a system.
Navigation though requires a multitude of sensors and expensive equipments in order to be carried out in a safe and intelligent manner. These sensors include LiDARs (Velodyne), cameras (RGB or RGBD), odometers, barometers, systems for inertial navigation such as the accelerometers, gyroscopes, systems for global localization, i.e., **Global Navigation Satellite Systems (GNNS)**, such as the GPS or GLONASS, infrared sensors, ultrasound sensors and more. All these sensors need to be integrated in an orderly manner, through sensor fusion (e.g., Kalman Filter) and then use computationally expensive algorithms such as SLAM (Simultaneous Localization And Mapping) to carry out navigation, therefore install high performance hardware on-board of the robot/UAG.
Autonomous Navigation therefore has a massive entry barrier given by costly sensors and integration of all these systems. Testing has always been carried out in controlled environments, so that faults aren't as damaging, however, the problem becomes of deeper complexity when it's carried out in extreme or unstructured outdoor environments. The amount of variability in the different scenarios, in highly cluttered environments, is far more complex than indoor or controlled settings.

The high costs related to autonomous navigation in real-world scenarios have lead to a scarcity of real-world data for Robots/UGVs, which is of critical importance for on-site real world applications. By real-world data we mean robots or vehicles which are able to adapt to a variety of situations, but this requires the vehicle to be deployed in harsh environments, where the vehicle might get damaged, and perform very large amounts of training.
However, applications in which real-world training is required have become of major interest, especially in the recent years, where there's been a growing research in the fields of planetary exploration, rescue tasks or precision agriculture.

To this end, Machine Learning and, in particular, the branch of Reinforcement Learning have proven to be of crucial importance in this context. Reinforcement Learning's strength relies on the ability to generalize control-strategies in a variety of scenarios.

Training of Agents can be performed on simulated environments, where breakdowns aren't costly, and multiple iterations of training can be performed in order to achieve control-strategies which are able to take the best actions whatever situation the Robot/UGV might end up in.

In order to capture as many scenarios as possible, and simulate highly cluttured environments, MIDGARD was developed. MIDGARD comes with four different natural environments, i.e., Meadow, Forest, Volcanic and Glacier.
A demonstration of the variable scene generation capabilities of MIDGARD are presented in the following figure:

<p align="center">
  <img src="https://github.com/Tatonta/Master_Thesis/blob/main/src/figures/MIDGARD_scenarios.png" scale=0.3/>
</p>

## Configurations
MIDGARD is a configurable simulation platform. The novelty of the simulator is that it's capable of changing the different environments by just changing the settings through an extensive set of APIs.
- OpenAI Gym-complaint Python frontend:  the simulator supports agent interaction through a standard OpenAI Gym environment, which has been previously mentioned in previous examples for training agents in simulated environments. It’s the standard commonly used for training and testing Reinforcement Learning algorithms in Python.
- Synchronous or asynchronous: The platform allows the AI agent to run both synchronously and asynchronously with respect to simulation time.
- Configurable simulation speed: Simulation speed is configurable and adjustable to requirements, allowing for faster training and predictable frame rate.
- Agent API: Provides flexible and customizable agent interaction, supporting the definition of user-defined control actions inside the simulation environment.
- Sensors API: Allows to define the set of sensors available to the agent (e.g., RGB camera, depth, semantic segmentation, instance segmentation, GPS-like location) and acquire observations and measurements.
## Scene Generation
In MIDGARD landscapes are generated procedurally, and there are 4 different types of scenarios, each one with it's own difficulty determined by the amount of obstacles which lie between the agent and the target, such as puddles, rocks, branches, etc...
The procedural generation is done by partitioning the landscape into a number of cells controlled by the difficulty level.
The effect of the difficulty on the obstacle density is seen in the following figure:
<p align="center">
  <img src="https://github.com/Tatonta/Master_Thesis/blob/main/src/figures/MIDGARD_clutteredenv.png" />
</p>

To run the different algorithms, run the command:
```
// For DQN
python3 trainDQN
// For PPO
python3 train_v2
// For SAC
python3 trainSAC
```
## PPO in Meadow and Glacier scenes
Below is presented a video of the PPO algorithm trained on the Meadow landscape with Curriculum learning and the Glacier landscape (with fixed difficulty = 0.3)

<div align="center">
  <a href="https://www.youtube.com/watch?v=Mn11uPtTb9Y"><img src="https://img.youtube.com/vi/Mn11uPtTb9Y/hqdefault.jpg" alt="IMAGE ALT TEXT"></a>
</div>
