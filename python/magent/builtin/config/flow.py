""" Only one """

import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": True})
    #cfg.set({"embedding_size": 10})
         
    agent = cfg.register_agent_type(
        name="agent",
        attr={'width': 1, 'length': 1, 'hp': 15, 'speed': 2,
              'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(1),
              'damage': 6, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              'attack_in_group': 1})

    g0 = cfg.add_group(agent)

    a = gw.AgentSymbol(g0, index='any')

    # reward shaping to encourage attack
    cfg.add_reward_rule(gw.Event(a, 'attack', a), receiver=a, value=-0.5)
    # end zone
    cfg.add_reward_rule(gw.Event(a, 'in', ((87, 47), (96, 73))), receiver=a, value=0.5)
    # start zone to encourage the agents to go right
    cfg.add_reward_rule( gw.Event(a, 'in', ((0, 38), (19, 99))), receiver=a, value=-0.5)


    return cfg
