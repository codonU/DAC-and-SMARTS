from starter_kit.train_example.utils.continuous_space import agent_spec
from pathlib import Path

scenario_path = [
    './dataset_public/all_loop/all_loop_a'
    # './dataset_public/intersection_loop/its_a',
    # './dataset_public/merge_loop/merge_a',s
    # './dataset_public/mixed_loop/its_merge_a',
    # './dataset_public/mixed_loop/roundabout_its_a',
    # './dataset_public/mixed_loop/roundabout_merge_a',
    # './dataset_public/roundabout_loop/roundabout_a',
    # './dataset_public/sharp_loop/sharploop_a',
    # './dataset_public/simple_loop/simpleloop_a',
]
scenario_path = Path(scenario_path[0]).absolute()
print(scenario_path)
games = [
    {
        'name': "smarts.env:hiway-v0",
        'scenario_path': scenario_path,
        'agent_spec': agent_spec,
        # 'headless': False,
        'visdom': False,
        'AGENT_ID': "AGENT-007",
    }
]

