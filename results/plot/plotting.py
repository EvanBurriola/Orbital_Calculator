import os
import sys
from glob import glob
import matplotlib.pyplot as plt

# Create directory for corresponding model
def is_dir_exist(file_id):
  isExist = os.path.exists(file_id)
  if not isExist:
    os.makedirs(file_id)


def process_files(file_list, plot_function, id):
    # Convert file type - 'ValueError: txt is not supported'
    for file in file_list:
        file_wo_extension, _ = os.path.splitext(file)
        os.rename(file, file_wo_extension)
        plot_function(file_wo_extension, id)


def plot_reward(file, id):
  file_name = os.path.basename(file)
  with open(file) as f:
    x  = [float(line.strip()) for line in f]
    plt.plot(x)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(file_name)
    # plt.show()
    plt.savefig(id+'/'+file_name)
    os.rename(file, file+'.txt')


def plot_state(file, id):
  file_name = os.path.basename(file)
  with open(file, 'r') as f:
    ep_num = f.readline()
    
    a = []
    e_x = []
    e_y = []
    h_x = []
    h_y = []
    for line in f:
      values = line.strip().split(',')
      a.append(float(values[0]))
      e_x.append(float(values[1]))
      e_y.append(float(values[2]))
      h_x.append(float(values[3]))
      h_y.append(float(values[4]))
    
    plt.clf()
    plt.subplot(3, 2, 1)
    plt.plot(a)
    plt.ylabel('a')
    
    plt.subplot(3, 2, 2)
    plt.plot(e_x)
    plt.ylabel('e_x')
    
    plt.subplot(3, 2, 3)
    plt.plot(e_y)
    plt.ylabel('e_y')
    
    plt.subplot(3, 2, 4)
    plt.plot(h_x)
    plt.ylabel('h_x')
    plt.xlabel('Iteration')
    
    plt.subplot(3, 2, 5)
    plt.plot(h_y)
    plt.ylabel('h_y')
    plt.xlabel('Iteration')

    # plt.suptitle(file_name)
    plt.tight_layout()
    # plt.show()
    plt.savefig(id+'/'+file_name)
    os.rename(file, file+'.txt')


def plot_action(file, id):
  file_name = os.path.basename(file)
  with open(file, 'r') as f:
    ep_num = f.readline()
    
    thrust_mag = []
    radial = []
    circum = []
    normal = []
    
    for line in f:
      values = line.strip().split(',')
      radial.append(float(values[0]))
      circum.append(float(values[1]))
      normal.append(float(values[2]))
      thrust_mag.append(float(values[3]))
    
    plt.clf()
    plt.subplot(2, 2, 1)
    plt.plot(thrust_mag)
    plt.title('Thrust Magnitude (N)', fontsize=10)
    
    plt.subplot(2, 2, 2)
    plt.plot(radial)
    plt.title('Radial Thrust (R)', fontsize=10)
    
    plt.subplot(2, 2, 3)
    plt.plot(circum)
    plt.title('Circumferential Thrust (S)', fontsize=10)
    plt.xlabel('Iteration')
    
    plt.subplot(2, 2, 4)
    plt.plot(normal)
    plt.title('Normal Thrust (W)', fontsize=10)
    plt.xlabel('Iteration')
    
    # plt.suptitle(file_name)
    plt.tight_layout()
    # plt.show()
    plt.savefig(id+'/'+file_name)
    os.rename(file, file+'.txt')


def main():
    n = len(sys.argv)
    if(n < 2):
        print("Usage: python3 plot.py [model_id]")
        sys.exit()

    id = sys.argv[1]
    reward_file = glob(f'../reward/*{id}*_reward.txt')

    # Error checking to ensure that model id exist
    if not reward_file:
      print('Invalid model number.')
      return
    
    is_dir_exist(id)
    state_files = [file for file in glob(f'../state/*{id}*/**') if 'kepler' not in file]
    action_files = glob(f'../action/*{id}*/**')

    process_files(reward_file, plot_reward, id)
    process_files(state_files, plot_state, id)
    process_files(action_files, plot_action, id)


if __name__ == "__main__":
    main()
