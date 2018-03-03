"""
Train agents to gather food, test flux
"""

import argparse
import logging as log
import time

import magent
from magent.builtin.mx_model import DeepQNetwork as RLModel
# change this line to magent.builtin.tf_model to use tensorflow


def load_config(size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": size, "map_height": size})
    cfg.set({"minimap_mode": True})

    agent = cfg.register_agent_type(
        name="agent",
        attr={'width': 1, 'length': 1, 'hp': 15, 'speed': 2,
              'view_range': gw.CircleRange(7), 'attack_range': gw.CircleRange(1),
              'damage': 6, 'step_recover': 0,
              'step_reward': -0.01,  'dead_penalty': -1, 'attack_penalty': -0.1,
              'attack_in_group': 1})

    food = cfg.register_agent_type(
        name='food',
        attr={'width': 1, 'length': 1, 'hp': 25, 'speed': 0,
              'view_range': gw.CircleRange(1), 'attack_range': gw.CircleRange(0),
              'kill_reward': 5})
              
              
    g_f = cfg.add_group(food)
    g_s = cfg.add_group(agent)

    a = gw.AgentSymbol(g_s, index='any')
    b = gw.AgentSymbol(g_f, index='any')

    cfg.add_reward_rule(gw.Event(a, 'attack', b), receiver=a, value=0.5)
    cfg.add_reward_rule(gw.Event(a, 'attack', a), receiver=a, value=-0.5)
    
    return cfg


def generate_map(env, map_size, food_handle, handles):
    center_x, center_y = map_size // 2, map_size // 2
    
    #####
    pos = []
    
    def add_full_square(pos, startx, starty, endx, endy, step=1):
        for i in range(startx, endx+1, step):
            for j in range(starty, endy+1, step):
                pos.append([i,j])
                
    add_full_square(pos, 0, 0, 19, 38)
    add_full_square(pos, 19, 0, 35, 64)#19, 64)
    add_full_square(pos, 35, 0, 80, 40)
    
    add_full_square(pos, 43, 47, 65, 71)
    add_full_square(pos, 65, 47, 80, 53)
    
    
    
    add_full_square(pos, 70, 58, 80, 75)
    
    add_full_square(pos, 19, 80, 80, 99)
    
    
    env.add_walls(pos=pos, method="custom")
    
    # Agent
    pos = []
    add_full_square(pos, 3, 57, 15, 86, step=2)
    add_full_square(pos, 4, 58, 16, 87, step=4)
    env.add_agents(handles[0], method="custom", pos=pos)
    
    # Food
    pos =[]
    add_full_square(pos, 85, 42, 96, 78)
    add_full_square(pos, 28, 67, 28, 77, step=2)
    env.add_agents(food_handle, method="custom", pos=pos)
    
    

    
def play_a_round(env, map_size, food_handle, handles, models, train_id=-1,
                 print_every=10, record=False, render=False, eps=None):
    env.reset()
    generate_map(env, map_size, food_handle, handles)

    step_ct = 0
    total_reward = 0
    done = False

    pos_reward_ct = set()

    n = len(handles)
    obs  = [None for _ in range(n)]
    ids  = [None for _ in range(n)]
    acts = [None for _ in range(n)]
    nums = [env.get_num(handle) for handle in handles]
    sample_buffer = magent.utility.EpisodesBuffer(capacity=5000)

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(handles[i])
            ids[i] = env.get_agent_id(handles[i])
            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps)
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample
        rewards = env.get_reward(handles[train_id])
        step_reward = 0
        if train_id != -1:
            alives  = env.get_alive(handles[train_id])
            total_reward += sum(rewards)
            sample_buffer.record_step(ids[train_id], obs[train_id], acts[train_id], rewards, alives)
            step_reward = sum(rewards)

        # render
        if render:
            env.render()

        for id, r in zip(ids[0], rewards):
            if r > 0.05 and id not in pos_reward_ct:
                pos_reward_ct.add(id)

        # clear dead agents
        env.clear_dead()

        # stats info
        for i in range(n):
            nums[i] = env.get_num(handles[i])
        food_num = env.get_num(food_handle)

        if step_ct % print_every == 0:
            print("step %3d,  train %d,  num %s,  reward %.2f,  total_reward: %.2f, non_zero: %d" %
                  (step_ct, train_id, [food_num] + nums, step_reward, total_reward, len(pos_reward_ct)))
        step_ct += 1

        if step_ct > 350:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    if record:
        with open("reward-hunger.txt", "a") as fout:
            fout.write(str(nums[0]) + "\n")

    # train
    total_loss = value = 0
    if train_id != -1:
        print("===== train =====")
        start_time = time.time()
        total_loss, value = models[train_id].train(sample_buffer, print_every=250)
        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return total_loss, total_reward, value, len(pos_reward_ct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=1500)
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--map_size", type=int, default=100)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--name", type=str, default="gather")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    # set logger
    log.basicConfig(level=log.INFO, filename=args.name + '.log')
    console = log.StreamHandler()
    console.setLevel(log.INFO)
    log.getLogger('').addHandler(console)

    # init env
    env = magent.GridWorld(load_config(size=100))#args.map_size))
    env.set_render_dir("build/render")

    handles = env.get_handles()
    food_handle = handles[0]
    player_handles = handles[1:]

    # sample eval observation set
    eval_obs = None
    if args.eval:
        print("sample eval set...")
        env.reset()
        generate_map(env, args.map_size, food_handle, player_handles)
        eval_obs = magent.utility.sample_observation(env, player_handles, 0, 2048, 500)

    # load models
    models = [
        RLModel(env, player_handles[0], args.name,
                batch_size=512, memory_size=2 ** 19, target_update=1000,
                train_freq=4, eval_obs=eval_obs)
    ]

    # load saved model
    save_dir = "save_model"
    if args.load_from is not None:
        start_from = args.load_from
        print("load models...")
        for model in models:
            model.load(save_dir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print('view_space', env.get_view_space(player_handles[0]))
    print('feature_space', env.get_feature_space(player_handles[0]))
    print('view2attack', env.get_view2attack(player_handles[0]))

    if args.record:
        for k in range(4, 999 + 5, 5):
            eps = 0
            for model in models:
                model.load(save_dir, start_from)
                play_a_round(env, args.map_size, food_handle, player_handles, models,
                             -1, record=True, render=False,
                             print_every=args.print_every, eps=eps)
    else:
        # play
        start = time.time()
        train_id = 0 if args.train else -1
        for k in range(start_from, start_from + args.n_round):
            tic = time.time()
            eps = magent.utility.piecewise_decay(k, [0, 400, 1000], [1.0, 0.2, 0.05]) if not args.greedy else 0
            loss, reward, value, pos_reward_ct = \
                    play_a_round(env, args.map_size, food_handle, player_handles, models,
                                 train_id, record=False,
                                 render=args.render or (k+1) % args.render_every == 0,
                                 print_every=args.print_every, eps=eps)
            log.info("round %d\t loss: %.3f\t reward: %.2f\t value: %.3f\t pos_reward_ct: %d"
                     % (k, loss, reward, value, pos_reward_ct))
            print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

            if (k + 1) % args.save_every == 0 and args.train:
                print("save models...")
                for model in models:
                    model.save(save_dir, k)