""" Only one """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    cfg.set({"embedding_size": 10})
         
    small = cfg.register_agent_type(
        "small",
        {'width': 1, 'length': 1, 'hp': 10, 'speed': 2,
         'view_range': gw.CircleRange(6), 'attack_range': gw.CircleRange(2),
         'damage': 0, 'step_recover': 0,

         'step_reward': -0.005,  'kill_reward': 0, 'dead_penalty': 0, 'attack_penalty': -0.1,
         })

    g0 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index='any')

    # reward shaping to encourage interaction
    cfg.add_reward_rule(gw.Event(a, 'attack', a), receiver=a, value=0.5)
    

    return cfg
