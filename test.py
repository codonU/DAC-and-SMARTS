import dm_control2gym


# Load one task:
domain = 'fish'
task = 'downleft'
# env = suite.load(domain_name="fish", task_name="downleft")
env = dm_control2gym.make(domain_name=domain, task_name=task)
# Iterate over a task set:
# for domain_name, task_name in suite.BENCHMARKING:
#   env = suite.load(domain_name, task_name)
#   print(domain_name, task_name)
# Step through an episode and print out reward, discount and observation.
# action_spec = env.action_spec()
# time_step = env.reset()
# while not time_step.last():
#   action = np.random.uniform(action_spec.minimum,
#                              action_spec.maximum,
#                              size=action_spec.shape)
#   time_step = env.step(action)
  # print(time_step.reward, time_step.discount, time_step.observation)
