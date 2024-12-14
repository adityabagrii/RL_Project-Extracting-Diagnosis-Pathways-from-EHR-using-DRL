# Extracting diagnosis pathways from EHR using Deep Reinforcement Learning

## Motivation and Problem Statement:
Today's state of the art Machine Learning classifiers can easily classify and give us the endpoint of the diagonsis of the output, but in the field of Healthcare the pathway taken to reach that specific end-point holds high importance. Through this project I have tried to build a Deep Reinforcement Learning model (DQN - Deep Q Learning) to both predict the endpoint of the diagnosis as well as also giving us the pathway followed to reach the endpoint.

## Dataset:
The dataset used in this model is taken from the research paper [Extracting Diagnosis Pathways from Electronic Health Records Using Deep Reinforcement Learning - Lillian Muyama, Antoine Neuraz, Adrien Coulet (Submitted on 10 May 2023)](https://arxiv.org/abs/2305.06295). They synthsised this dataset for the disease Anemia by consulting domain experts and figuring out the factors leading to the disease further used a Decision tree model to generate the said model, then used a 80-20 split to split it into training and testing sets.

## Decision Problem:
- **State-Space:** The state of the problem is represented as a observation space using an array consisting a of some columns, where if a column has a value other than -1, that means the value of this column/feature is known otherwise is yet to be observed.
- **Action-Space:** There are two types of actions. Diagnosis Actions -- These actions give the final diagnosis based on the trajectory and are thereby terminating actions. Feature Actions -- These actions query the environment about the value of a feature and in return observe the value of the queried feature in the next observed state, these are non-terminating actions.
- **Rewards:**
    - For a Diagnosis Action - +5 for a correct diagnosis and -100 for incorrect as incorrect diagnosis can be fatal in the healthcare department.
    - For a Feature Action - +1 for a feature query to promote the agent to explore the available data options.
    - For a repeating query - -1000 to strongly discourage the use of repeated query.

## Implementaion of DQN:
The DQN network is implemented using the python library stable-baselines3. It is trained for more than 2x10e7 training steps on the training datasets. The test split of the dataset has been used to check the performance of the model and calculate its accuracy on the test data.