import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import _ones_like_dispatcher
import seaborn as sns
from scipy.stats import poisson

MAX_CARS = 20

MAX_MOVE_CARS = 5

AVG_RENT_A = 3
AVG_RENT_B = 4
AVG_RETURN_A = 3
AVG_RETURN_B = 2

DISCOUNT = 0.9

RENT_BONUS = 10

MOVE_COST = 2

actions = np.arange(-MAX_MOVE_CARS, MAX_MOVE_CARS + 1)

POISSON_UPPER_BOUND = 11

poisson_cahce = dict()

def poisson_probability(n, lam):
    global poisson_cahce
    key = n * 10 + lam
    if key not in poisson_cahce:
        poisson_cahce[key] = poisson.pmf(n,lam)
    return poisson_cahce[key]

# 计算状态价值
def cal_v(state, action, state_value, returned_cars):
    """
    @state：每个地点的车辆数
    @action：移动
    @state_value：状态价值矩阵
    @returned_cars：还车数目
    """
    returns = 0.0
    returns -= MOVE_COST * abs(action)
    
    NUM_OF_CARS_A = min(state[0] - action, MAX_CARS)
    NUM_OF_CARS_B = min(state[1] + action, MAX_CARS)

    # 遍历两地全部可能概率下租车数目请求
    for rent_req_A in range(POISSON_UPPER_BOUND):
        for rent_req_B in range(POISSON_UPPER_BOUND):
            prob = poisson_probability(rent_req_A, AVG_RENT_A) *\
                poisson_probability(rent_req_B,AVG_RENT_B)
            
            num_of_cars_A = NUM_OF_CARS_A
            num_of_cars_B = NUM_OF_CARS_B

            valid_rent_A = min(num_of_cars_A, rent_req_A)
            valid_rent_B = min(num_of_cars_B, rent_req_B)

            reward = (valid_rent_A + valid_rent_B) * RENT_BONUS
            num_of_cars_A -= valid_rent_A
            num_of_cars_B -= valid_rent_B

            # 如果还车的数目为泊松分布的均值
            if returned_cars:
                returned_cars_A = AVG_RETURN_A
                returned_cars_B = AVG_RETURN_B

                num_of_cars_A = min(num_of_cars_A + returned_cars_A, MAX_CARS)
                num_of_cars_B = min(num_of_cars_B + returned_cars_B, MAX_CARS)

                # 策略评估
                returns += prob * (reward + DISCOUNT * state_value[num_of_cars_A,num_of_cars_B])
            
            # 计算所有泊松概率分布下的还车空间
            else:
                for returned_cars_A in range(POISSON_UPPER_BOUND):
                    for returned_cars_B in range(POISSON_UPPER_BOUND):
                        prob_return = poisson_probability(returned_cars_A, AVG_RETURN_A) * poisson_probability( returned_cars_B, AVG_RETURN_B)
                        num_of_cars_A_ = min(num_of_cars_A + returned_cars_A, MAX_CARS)
                        num_of_cars_B_ = min(num_of_cars_B + returned_cars_B, MAX_CARS)
                        # 联合概率
                        prob_ = prob_return * prob
                        returns += prob_ * (reward + DISCOUNT) * state_value[num_of_cars_A_, num_of_cars_B_]
    return returns                    


def draw(constant_returned_cars = True):
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy = np.zeros(value.shape, dtype=np.int)
    iterations = 0

    # 准备画布，准备多个子图
    _, axes = plt.subplots(2, 3, figsize=(40, 20))
    # 调整子图间距
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    # 将子图形成1 * 6 的列表
    axes = axes.flatten()
    while True:
        # 使用seaborn的heatmap作图
        fig = sns.heatmap(np.flipud(policy), cmap="YlGnBu", ax=axes[iterations])

        fig.set_ylabel('# cars at A',fontsize = 30)
        fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
        fig.set_xlabel('# cars at B',fontsize = 30)
        fig.set_title('policy{}'.format(iterations), fontsize = 30)

        while True:
            old_value = value.copy()
            for i in range(MAX_CARS + 1):
                for j in range(MAX_CARS + 1):
                    # 更新v(s)
                    new_state_value = cal_v([i,j], policy[i,j], value, constant_returned_cars)

                    value[i,j] = new_state_value
            max_value_change = abs(old_value - value).max()
            print('max value change {}'.format(max_value_change))
            if(max_value_change < 1e-4):
                break
        
        policy_stable = True
        # i,j 为 A、B两地的现有车辆
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                old_action= policy[i,j]
                action_returns = []

                for action in actions:
                    if(0 <= action <= i) or (-j <=action <= 0):
                        action_returns.append(cal_v([i,j], action, value, constant_returned_cars))
                    else:
                        action_returns.append(-np.inf)

                new_action = actions[np.argmax(action_returns)]

                policy[i,j] = new_action

                if policy_stable and old_action != new_action:
                    policy_stable = False
        
        print('policy stable{}'.format(policy_stable))
        if policy_stable:
            fig = sns.heatmap(np.flipud(value), cmap="YlGnBu", ax=axes[-1])
            fig.set_ylabel('# cars at first location', fontsize=30)
            fig.set_yticks(list(reversed(range(MAX_CARS + 1))))
            fig.set_xlabel('# cars at second location', fontsize=30)
            fig.set_title('optimal value', fontsize=30)
            break

        iterations += 1

    plt.savefig('C:/INFO\RL/assignment/2_Jack/fig.png')
    plt.close()


if __name__ == '__main__':
    draw()

                    

