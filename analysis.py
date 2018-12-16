paths = [
    './experiments/rl_model_unit_survive_epoch_10_reward_distance.pth',
    './experiments/rl_model_unit_survive_epoch_10_reward_survival.pth',
    './experiments/rl_model_unit_survive_epoch_10_reward_score.pth',
    './experiments/rl_model_unit_survive_epoch_30_reward_distance.pth',
    './experiments/rl_model_unit_survive_epoch_30_reward_survival.pth',
    './experiments/rl_model_unit_survive_epoch_30_reward_score.pth',
    './experiments/rl_model_unit_survive_epoch_50_reward_distance.pth',
    './experiments/rl_model_unit_survive_epoch_50_reward_survival.pth',
    './experiments/rl_model_unit_survive_epoch_50_reward_score.pth'

]

# save_model_path = './experiments/rl_model_unit_survive_epoch_10_reward_distance.pth'
# save_model_path = './experiments/rl_model_unit_survive_epoch_10_reward_survival.pth'
# save_model_path = './experiments/rl_model_unit_survive_epoch_10_reward_score.pth'

# save_model_path = './experiments/rl_model_unit_survive_epoch_30_reward_distance.pth'
# save_model_path = './experiments/rl_model_unit_survive_epoch_30_reward_survival.pth'
# save_model_path = './experiments/rl_model_unit_survive_epoch_30_reward_score.pth'

# save_model_path = './experiments/rl_model_unit_survive_epoch_50_reward_distance.pth'
# save_model_path = './experiments/rl_model_unit_survive_epoch_50_reward_survival.pth'
# save_model_path = './experiments/rl_model_unit_survive_epoch_50_reward_score.pth'

import matplotlib.pyplot as plt
import torch

for path in paths:
    fitnesses = torch.load(path, 'cpu')['fitnesses']
    highest = max(fitnesses)

    print('{} highest'.format(path), highest)

    plt.figure()
    plt.subplot(111)
    # plt.title('generation{}'.format())
    plt.plot(fitnesses)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.savefig(path.replace('.pth', '.png'))
    plt.show()
