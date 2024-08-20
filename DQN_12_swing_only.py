import numpy as np
import random
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import os
import time
import datetime
import tracemalloc
import pandas as pd
import itertools
import psutil
import gc
from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from math import log
from functools import partial

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 각 GPU에 대해 메모리 성장을 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("hi")
    except RuntimeError as e:
        print(e)
from memory_profiler import profile
import sys
tracemalloc.start()
process = psutil.Process()
start_memory = process.memory_info().rss

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        # self.epsilon_decay = 0.95 # 100
        # self.epsilon_decay = 0.984 # 300
        # self.epsilon_decay = 0.991 # 500
        # self.epsilon_decay = 0.9935 # 700
        # self.epsilon_decay = 0.995 #900
        self.epsilon_decay = 0.9975  # 1900
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.trainable = True
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))  # inpuy size, layer 수정 필요
        model.add(Dense(128, activation='relu'))
        # model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        # model.trainable = True
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            action = action % self.action_size
            return action
        act_values = self.model.predict(state,verbose=0)
        return np.argmax(act_values[0])

    def replay(self, agent, batch_size):

        minibatch = random.sample(self.memory, batch_size)

        states = np.array([state for state, _, _, _, _ in minibatch])
        states = np.squeeze(states)
        next_states = np.array([next_state for _, _, _, next_state, _ in minibatch])
        next_states = np.squeeze(next_states)

        next_qs = agent.target_model.predict(next_states,batch_size=batch_size,verbose=0)
        current_qs = self.target_model.predict(states,batch_size=batch_size,verbose=0)


        for i, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                current_qs[i][action] = reward
            else:
                current_qs[i][action] = reward + self.gamma * np.amax(next_qs[i])
        # Using fit to train the model on the entire minibatch at once
        loss_history = self.model.fit(states, current_qs, epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon: ", self.epsilon)


        return loss_history


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
        return


# 환경 설정, 에이전트 초기화
state_size = 216
# action_size = 4
action_size = 3
agents = []

for num in range(24):
    agents.append(DQNAgent(state_size, action_size))
num_agent = len(agents)
num_episodes = 2000
batch_size = 32


# 기지국의 위치 정보
BS_file_path = 'D:/12국소_5.xlsx'  ## 선택된 BS의 map을 넣도록 파일을 수정해야함.
BS_df = pd.read_excel(BS_file_path)
BS_coor = BS_df[['x', 'y', 'total_height']].to_numpy()

# waypoint와 관찰점에 대한 정보
waypoint_file_path = 'D:/service_line.xlsx'
waypoint_df = pd.read_excel(waypoint_file_path)
waypoint_coor = waypoint_df[['x', 'y', 'num_of_measure_point']].to_numpy()

# setting value
# BS_pos = BS_coor
BS_pos = BS_coor[::2]
print(BS_pos)
num_waypoint = len(waypoint_coor) - 2
num_BS = len(BS_pos)
z_point = 300  # [m]

# 전체 coordinates 배열을 num_waypoint 길이로 초기화
all_coordinates = []

for idx in range(num_waypoint):
    coordinates = np.zeros((0, 2))  # 빈 배열로 초기화
    if idx < num_waypoint - 1:
        start_x = waypoint_coor[idx][0]
        start_y = waypoint_coor[idx][1]
        end_x = waypoint_coor[idx + 1][0]
        end_y = waypoint_coor[idx + 1][1]
        num_points = int(waypoint_coor[idx][2])
    else:  # 마지막 waypoint와 시작 waypoint간의 측정점 계산
        start_x = waypoint_coor[idx][0]
        start_y = waypoint_coor[idx][1]
        end_x = waypoint_coor[0][0]
        end_y = waypoint_coor[0][1]
        num_points = int(waypoint_coor[idx][2])

    # linspace를 사용하여 각 축에 대한 좌표 생성, 시작점과 종점 제외
    x_values = np.linspace(start_x, end_x, num_points)[1:-1]
    y_values = np.linspace(start_y, end_y, num_points)[1:-1]

    # 시작점 좌표를 추가
    coordinates = np.vstack((coordinates, np.array([start_x, start_y])))

    # 결과 좌표를 쌓아서 저장
    coordinate = np.column_stack((x_values, y_values))
    coordinates = np.vstack((coordinates, coordinate))

    all_coordinates.append(coordinates)

# 배열을 리스트에서 최종적으로 하나의 ndarray로 변환
all_coordinates = np.array(all_coordinates, dtype=object)
measure_point = np.concatenate(all_coordinates, axis=0)
num_measure_point = len(measure_point)
# print("hello")
# print(num_measure_point)
swings = [75, 285, 225, 45, 165, 10, 170, 0, 180, 335, 165, 305, 130, 300, 
130, 330, 150, 315, 150, 15, 145, 315, 135, 270]
tilts = [29, 28, 28, 51, 29, 38, 38, 32, 32, 39, 39, 47, 47, 34, 
38, 39, 34, 43, 39, 27, 43, 27, 27, 27]
swings_fix = [75, 285, 225, 45, 165, 10, 170, 0, 180, 335, 165, 305, 130, 300, 
130, 330, 150, 315, 150, 15, 145, 315, 135, 270]
tilts_fix = [29, 28, 28, 51, 29, 38, 38, 32, 32, 39, 39, 47, 47, 34, 
38, 39, 34, 43, 39, 27, 43, 27, 27, 27]
comb_st = [swings.copy(),tilts.copy()]
carrier_freq = 0.866  # [GHz]
RSRP_threshold = -83  # [dBm]

transmit_power = 49  # [dBm]
BW = 15000  # [Hz]
noise = -174  # [dBm/Hz]

antenna_max_gain_dBi = 11  # [dBi]
cable_loss_dB = 1  # [dB]
UE_noise_figure_dB = 9  # [dB]
AV_penetration_dB = 8.5  # [dB]

df = pd.read_excel('D:/antenna_pattern.xlsx', sheet_name='Switched Speaker 3.5GHz')
# third_column_name = df.columns[2]
# df.drop(columns=[third_column_name], inplace=True)
# df.set_index(df.columns[0], inplace=True)

#저장경로 설정
base_path = "D:/HW/real_line_DQN/0818_DQN_BS{}_tilt_+++".format(num_BS)

# 필요한 폴더 경로 생성
losses_folder = os.path.join(base_path, "losses")
rewards_folder = os.path.join(base_path, "rewards")


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_folder(base_path)  # '240412' 폴더 생성
create_folder(losses_folder)
create_folder(rewards_folder)

def calculate_theta(point_x, point_y, point_z, pos_x, pos_y, pos_z):
    x_distance = point_x - pos_x
    y_distance = point_y - pos_y
    z_distance = point_z - pos_z

    theta_rad = math.atan2(z_distance, math.sqrt(x_distance ** 2 + y_distance ** 2))  ## 이건 플마 180아니고 90도인데 얘도 동일하게 atan2 써도되나?

    # 각도를 degree로 변환
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def calculate_phi(point_x, point_y, pos_x, pos_y):
    x_distance = point_x - pos_x
    y_distance = point_y - pos_y
    # 기지국과 point 사이의 각도 계산 [radian]
    phi_rad = math.atan2(y_distance, x_distance)

    # 각도를 degree로 변환
    phi_deg = np.degrees(phi_rad)
    if phi_deg < 0:
        phi_deg += 360
    return phi_deg



def calculate_swing_antenna_gain(phi):  # 안테나 패턴 값 계산

    # 빔 지향 각도와 수직 방향 사이의 각도 차이를 계산
    phi_diff = phi
    phi_diff = round(phi_diff)
    # phi_diff *= -1

    if phi_diff < 0:
        phi_diff = 360 + phi_diff

    # matching_rows = df[df.iloc[:, 0] == phi_diff]
    # second_column_values = matching_rows.iloc[:, 2].tolist()
    #
    try:
        # 인덱스를 사용하여 값 조회
        second_column_values = df.loc[phi_diff].iloc[-1]
    except KeyError:
        # KeyError가 발생하면 에러를 발생시킴
        raise KeyError(f"Phi difference {phi_diff} not found in DataFrame index.")
    second_column_values = [second_column_values]
    # second_column_values = df.loc[phi_diff].tolist()
    # print(second_column_values)
    return second_column_values


def calculate_tilt_antenna_gain(theta):  # 안테나 패턴 값 계산

    # 빔 지향 각도와 수직 방향 사이의 각도 차이를 계산
    theta_diff = theta
    theta_diff = round(theta_diff)
    # theta_diff *= -1

    if theta_diff < 0:
        theta_diff = 360 + theta_diff

    # matching_rows = df[df.iloc[:, 0] == theta_diff]
    # second_column_values = matching_rows.iloc[:, 1].tolist()

    try:
        # 인덱스를 사용하여 값 조회
        second_column_values = df.loc[theta_diff].iloc[-2]
    except KeyError:
        # KeyError가 발생하면 에러를 발생시킴
        raise KeyError(f"Phi difference {theta_diff} not found in DataFrame index.")
    second_column_values = [second_column_values]
    # second_column_values = df.loc[theta_diff].tolist()
    # print(second_column_values)
    return second_column_values


def auto_tilt(BS_idx, swing, path_coordinates=waypoint_coor):
    a, b, z_c = BS_pos[BS_idx]
    # 스윙 각도에 따른 기울기 계산
    slope = np.tan(np.radians(swing))
    
    # 스윙 각도에 따른 직선 방정식: y = slope * (x - a) + b
    closest_point = None
    min_distance = float('inf')
    
    for i in range(len(path_coordinates) - 1):
        # 경로의 두 점 (x1, y1), (x2, y2) 가져오기
        x1= path_coordinates[i][0]
        y1= path_coordinates[i][1]
        x2= path_coordinates[i+1][0]
        y2= path_coordinates[i+1][1]
        # 경로 구간의 직선 방정식 y = m * x + c 계산
        if x2 != x1:  # x 좌표가 다를 경우에만 기울기 계산
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            # 스윙 직선과 경로 직선의 교차점 계산
            x_intersect = (c - b + slope * a) / (slope - m)
            y_intersect = slope * (x_intersect - a) + b
            
            # 교차점이 경로 구간 내에 있는지 확인
            if min(x1, x2) <= x_intersect <= max(x1, x2) and min(y1, y2) <= y_intersect <= max(y1, y2):
                distance = np.sqrt((x_intersect - a) ** 2 + (y_intersect - b) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = (x_intersect, y_intersect)
                    print("min::::::::::::::::::",x1,y1,x2,y2)
        
        else:  # x 좌표가 같을 경우(수직선)
            x_intersect = x1
            y_intersect = slope * (x_intersect - a) + b
            
            if min(y1, y2) <= y_intersect <= max(y1, y2):
                distance = np.sqrt((x_intersect - a) ** 2 + (y_intersect - b) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = (x_intersect, y_intersect)
                    print("min::::::::::::::::::",x1,y1,x2,y2)

    if closest_point is not None:
        # closest_point는 (x, y) 이며, z_point를 사용하여 z좌표를 추가
        x, y = closest_point
        z = z_point

        # (a, b, c)와 (x, y, z) 사이의 tilt 각도 계산
        delta_x = x - a
        delta_y = y - b
        delta_z = z - z_c

        # 수평거리
        horizontal_distance = np.sqrt(delta_x**2 + delta_y**2)

        # tilt 각도 계산 (라디안 -> 도)
        tilt_angle = np.degrees(np.arctan2(delta_z, horizontal_distance))
        print(a,b)
        print(x,y)
        print("tilt",tilt_angle)
        tilt_angle = round(tilt_angle)
        return tilt_angle

    return False


def cart2sph_swingtilt_axis(measure_x, measure_y, measure_z, BS_x, BS_y, BS_z, swing, tilt):

    # 측정하려는 좌표를 안테나 위치만큼 평행이동
    x_m = measure_x - BS_x
    y_m = measure_y - BS_y
    z_m = measure_z - BS_z

    # 회전 각도 단위 변환 (도 -> 라디안)
    ang_s_rad = np.deg2rad(swing)
    ang_t_rad = np.deg2rad(tilt)

    # x'축 단위 벡터
    # 구면좌표계 -> 직교좌표계
    a_x, a_y, a_z = np.cos(ang_s_rad)* np.cos(ang_t_rad), np.cos(ang_t_rad) * np.sin(ang_s_rad), np.sin(ang_t_rad)
    a = np.array([a_x, a_y, a_z])

    # 기존 좌표계를 x'축으로 회전시키는 회전 행렬 계산
    u = a[0]
    v = a[1]
    w = a[2]

    R = np.array([
        [u, v, w],
        [-v / np.sqrt(u**2 + v**2), u / np.sqrt(u**2 + v**2), 0],
        [-u * w / np.sqrt(u**2 + v**2), -v * w / np.sqrt(u**2 + v**2), np.sqrt(u**2 + v**2)]
    ])

    # 카테시안 좌표를 x'축 기준으로 회전
    xyz_rotated = R @ np.array([x_m, y_m, z_m])

    # 회전된 좌표
    x_rot = xyz_rotated[0]
    y_rot = xyz_rotated[1]
    z_rot = xyz_rotated[2]

    # 구면 좌표 계산
    phi_dflt = np.arctan2(y_rot, x_rot)
    theta_dflt = np.arctan2(z_rot, np.sqrt(x_rot**2 + y_rot**2))

    phi = np.rad2deg(phi_dflt)
    theta = np.rad2deg(theta_dflt)

    r = np.sqrt(x_rot**2 + y_rot**2 + z_rot**2)

    return phi, theta, r
def calculate_pathloss(d, f):  # pathloss 계산 ~ cal_pathloss
    # d: 거리(m), f: 주파수(GHz)
    return 28.0 + 22 * np.log10(d) + 20 * np.log10(f)

#
# tilts = []
# for i in range(num_BS):
#     tilts.append(round(math.degrees(
#         math.atan2(z_point - BS_pos[i][2], 430))))  # BS_pos[i][2] = BS의 total height. th 105일 경우, 5000 / th 83일 경우 430



def cal_RSRP(BS_idx, swing1, swing2, tilt1, tilt2, measure_points):
    rsrp_values = []
    xpos, ypos, zpos = BS_pos[BS_idx]
    left_beam_swing, right_beam_swing = swing1, swing2
    left_beam_tilt, right_beam_tilt = tilt1, tilt2


    for coor in measure_points:
        xpoint, ypoint = coor

        left_phi,left_theta,d= cart2sph_swingtilt_axis(xpoint, ypoint, z_point, xpos, ypos, zpos, left_beam_swing, left_beam_tilt)
        right_phi,right_theta,_ = cart2sph_swingtilt_axis(xpoint, ypoint, z_point, xpos, ypos, zpos, right_beam_swing, right_beam_tilt)
        # print(left_phi,left_theta,d1)

        pathloss_db = calculate_pathloss(d, carrier_freq)

        # 두 swing이 off == BS off일 경우 아주 작은 값 저장
        if right_beam_swing == 360 and left_beam_swing == 360:
            total_Rx_power_dBm = -500

        else:
            swing_left_antenna_gain_db = calculate_swing_antenna_gain(left_phi)[0]
            swing_right_antenna_gain_db = calculate_swing_antenna_gain(right_phi)[0]

            tilt_left_antenna_gain_db = calculate_tilt_antenna_gain(left_theta)[0]
            tilt_right_antenna_gain_db = calculate_tilt_antenna_gain(right_theta)[0]

            total_left_antenna_gain_db = swing_left_antenna_gain_db + tilt_left_antenna_gain_db + antenna_max_gain_dBi
            total_right_antenna_gain_db = swing_right_antenna_gain_db + tilt_right_antenna_gain_db + antenna_max_gain_dBi

            left_Rx_singal_dbm = transmit_power + total_left_antenna_gain_db - pathloss_db - cable_loss_dB - UE_noise_figure_dB - AV_penetration_dB
            right_Rx_singal_dbm = transmit_power + total_right_antenna_gain_db - pathloss_db - cable_loss_dB - UE_noise_figure_dB - AV_penetration_dB
            left_Rx_singal_lin = 10 ** (left_Rx_singal_dbm / 10)
            right_Rx_singal_lin = 10 ** (right_Rx_singal_dbm / 10)
            total_Rx_power_lin = left_Rx_singal_lin + right_Rx_singal_lin
            total_Rx_power_dBm = 10 * np.log10(total_Rx_power_lin)

        RSRP_dBm = total_Rx_power_dBm - 24.77  # [dB]

        rsrp_values.append(10 ** (RSRP_dBm / 10))

    rsrp_values = np.reshape(rsrp_values, [1, len(measure_points)])  # nparr로 저장 후 리턴

    return rsrp_values  # lin


def cal_SINR(SUM_RSRP_lin, max_rsrp_lin, noise, BW):  # 받은 all rsrp는 lin scale. nparr 상태로 연산

    interference_lin = SUM_RSRP_lin - max_rsrp_lin
    noise_power = noise + 10 * np.log10(BW)  # dBm
    noise_power_lin = 10 ** (noise_power / 10)  # lin

    SINR_lin = max_rsrp_lin / (interference_lin + noise_power_lin)  ### lin scale서 계산
    SINR = 10 * np.log10(SINR_lin)  # log scale로 return

    return SINR

def calculate_reward(swing_tlt):
    reward_ = 0
    rsrp_values = []
    # for i in range(num_BS):
    # print(swing_tlt)
    for i in range(num_BS):
        rsrp_value = cal_RSRP(i, swing_tlt[0][2 * i], swing_tlt[0][2 * i + 1],
                              swing_tlt[1][2 * i], swing_tlt[1][2 * i + 1], measure_point)
        rsrp_values.append(rsrp_value)
    SUM_RSRP_lin = sum(rsrp_values)
    max_rsrp_lin = np.maximum.reduce(rsrp_values)
    max_rsrp_dbm = 10 * np.log10(max_rsrp_lin)
    max_rsrp_dbm = np.round(max_rsrp_dbm)
    medi = cal_SINR(SUM_RSRP_lin, max_rsrp_lin, noise, BW)
    medi = np.median(medi)
    cover = 100 * (np.count_nonzero(max_rsrp_dbm >= RSRP_threshold) / num_measure_point)
    # cover = 1 - hole/len(max_rsrps_lin)
    # cover = 100*cover
    # medi = np.median(SINR)
    # reward_ = 10 * medi
    # reward_ = round(reward_, 2)
    # medi = round(medi, 2)
    cover = round(cover, 2)
    if cover < 95:
        reward_ = 0
    else:
        reward_ = cover
    # if cover < 99:
    #     if cover>98:
    #         reward_ *= pow(cover/100,4)
    #         reward_ = round(reward_, 2)
    #     else:
    #         reward_ = 0
    # elif 100> cover > 99.2:
    #     cover = 100
    # return reward_, medi, cover
    return reward_, cover
b = [4,5,6,7,8]

def update_state(idx, selected_act):
    done = False
    idx_beam = idx%2
    idx_beams = idx//2
    if selected_act == 2:
        return to_one_hot(comb_st), done
    elif selected_act == 0:
        if swings[idx] > swings_fix[idx]-8:
            swings[idx] -= 2
            comb_st[0][idx] -= 2
            a_t = auto_tilt(idx_beams,swings[idx])
            if a_t is not False:
                tilts[idx] = a_t
                comb_st[1][idx] = a_t
        else:
            done = True
            return to_one_hot(comb_st), done
    elif selected_act == 1:
        if swings[idx] < swings_fix[idx]+8:
            swings[idx] += 2
            comb_st[0][idx] += 2
            a_t = auto_tilt(idx_beams,swings[idx])
            if a_t is not False:
                tilts[idx] = a_t
                comb_st[1][idx] = a_t
        else:
            done = True
            return to_one_hot(comb_st), done

    return to_one_hot(comb_st), done

def to_one_hot(array, s=swings_fix, num_classes=9):
    a = array[0]  # swing 값을 가져옴
    one_hot_a = np.zeros((len(a), num_classes))

    for i, val_a in enumerate(a):
        diff_a = val_a - s[i]

        if diff_a == -8:
            one_hot_a[i, 0] = 1
        elif diff_a == -6:
            one_hot_a[i, 1] = 1
        elif diff_a == -4:
            one_hot_a[i, 2] = 1
        elif diff_a == -2:
            one_hot_a[i, 3] = 1
        elif diff_a == 0:
            one_hot_a[i, 4] = 1
        elif diff_a == 2:
            one_hot_a[i, 5] = 1
        elif diff_a == 4:
            one_hot_a[i, 6] = 1
        elif diff_a == 6:
            one_hot_a[i, 7] = 1
        elif diff_a == 8:
            one_hot_a[i, 8] = 1

    # Flatten the one-hot encoded array and reshape to (1, -1)
    one_hot_flattened = one_hot_a.flatten().reshape(1, -1)
    return one_hot_flattened


def plot_reward(reward, episode_index):
    filename = os.path.join(rewards_folder, 'rewards_{}.txt'.format(episode_index))
    with open(filename, 'w') as file:
        for value in reward:
            file.write('{}\n'.format(value))

def plot_loss(loss, episode_index):
    filename = os.path.join(losses_folder, 'losses_{}.txt'.format(episode_index))
    with open(filename, 'w') as file:
        for value in loss:
            # 값이 리스트 안의 실수라고 가정하고, 리스트에서 실수를 추출하여 기록합니다.
            file.write('{}\n'.format(value[0]))

actions = [2 for _ in range(24)]

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss  # in bytes


start_memory = get_memory_usage()
start_time = time.time()

for e in range(num_episodes):
    next_state = np.zeros(num_agent + 1, dtype=object)
    state = np.zeros(num_agent + 1, dtype=object)
    done = [False] * (num_agent + 1)
    losses9 = []
    rewards = []

    swings = [75, 285, 225, 45, 165, 10, 170, 0, 180, 335, 165, 305, 130, 300, 
    130, 330, 150, 315, 150, 15, 145, 315, 135, 270]
    tilts = [29, 28, 28, 51, 29, 38, 38, 32, 32, 39, 39, 47, 47, 34, 
    38, 39, 34, 43, 39, 27, 43, 27, 27, 27]
    comb_st=[swings.copy(),tilts.copy()]

    state_temp = to_one_hot(comb_st)
    reward, _ = calculate_reward(comb_st)
    print("num_BS", num_BS)
    # timestep 루프 시작
    for tp in range(20):
        for k in range(num_agent):
            print("agent{} start: ".format(k), comb_st)
            state[k] = state_temp
            actions[k] = agents[k].act(state[k])
            next_state[k], done[k] = update_state(k, actions[k])

            reward_temp, cover = calculate_reward(comb_st)
            if done[k] == True:
                reward_now = -100
            else:
                reward_now = reward_temp - reward
            agents[k].remember(state[k], actions[k], reward_now, next_state[k], done[k])
            # print(state[k] , next_state[k])
            state_temp = next_state[k]

            if len(agents[k].memory) > batch_size:
                if k == num_agent-1:
                    hist1 = agents[k].replay(agents[0], batch_size)
                    # losses9.append([hist1])
                    losses9.append(hist1.history['loss'])
                else:
                    agents[k].replay(agents[k + 1], batch_size)
                gc.collect()

            if k == num_agent-1:
                rewards.append(reward_temp)
            reward = reward_temp

        # 타임스텝 루프 후 메모리 사용량 출력
        end_memory = get_memory_usage()
        total_memory_used = end_memory
        print(f"메모리 사용량: {total_memory_used / (1024 * 1024 * 1024):.2f} GB")

        # 현재 상태 출력
        print('Epi = {}. tp = {}. act = {}.  Reward = {}. swingtilt = {}. cover = {}% .'.format(e, tp, actions,reward,comb_st, cover,))
        plot_reward(rewards, e)
        plot_loss(losses9, e)
        if any(done):
            break

    for k in range(num_agent):
        agents[k].update_target_model()

    end_time = time.time()  # 코드 실행 종료 시간
    total_time = end_time - start_time  # 총 실행 시간 계산
    times = str(datetime.timedelta(seconds=total_time))  # 걸린시간 보기좋게 바꾸기
    short = times.split(".")[0]  # 초 단위 까지만

    print(f'Total Execution Time: {short} seconds')  # 총 실행 시간 출력