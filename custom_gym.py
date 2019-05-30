# Comp. Engr. Bruno Brandão Soares Martins

from scipy.misc import imresize
import gym
from gym.wrappers import Monitor
import numpy as np

# Para espaço de observação na forma de imagem e ações discretas!!!
# Bom para ATARI
class CustomGym:
    def __init__(self, game_name, skip_actions=4, num_frames=4, w=84, h=84, record=False):
        # game_name: nome do ambiente no OpenAI Gym
        # skip_actions: número de frames que uma ação será repetida
        # num_frames: número de frames que serão empilhados para entrar na rede
        # w: largura desejada da imagem
        # h: altura desejada da imagem

        self.env = gym.make(game_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h

        if record:
            self.env = Monitor(self.env, './videos_a3c/', video_callable=lambda episode_id: episode_id % 1 == 0)

        self.action_space = range(self.env.action_space.n)

        self.action_size = len(self.action_space)
        self.state = None
        self.game_name = game_name

    def preprocess(self, frame, is_start=False):
        # frame: frame a ser processado
        # is_start: se é, ou não, o início de um novo jogo, neste caso ele não
        # deverá ser empilhado com os anteriores

        # conversão para escala der cinza
        greyscale = frame.astype('float32').mean(axis=2)
        # tentar com np.uint8 para utilizar menos memória

        # ajustando o tamanho da imagem (downsampling) e normalização dos valores
        s = imresize(greyscale, (self.w, self.h)).astype('float32') * (1.0/255.0)

        # redimensionando para 4D com a primeira e a última dimensões sendo 1
        # para empilhá-los
        s = s.reshape(1, s.shape[0], s.shape[1], 1)

        # caso seja o início de um novo jogo o mesmo frame é copiado
        if is_start or self.state is None:
            self.state = np.repeat(s, self.num_frames, axis=3)
        else:
            self.state = np.append(s, self.state[:,:,:,:self.num_frames-1], axis=3)

        return self.state

    # renderiza o frame atual
    def render(self):
        self.env.render()

    # reseta o ambiente e retorna o estado
    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    # da um passo no ambiente com a ação escolhida 'skip_actions' vezes
    def step(self, action_index):
        action = self.action_space[action_index] # mapear ação
        acc_reward = 0 # recompensa acumulada apenas nos frames que serão pulados
        previous_frame = None
        for anything in range(self.skip_actions):
            frame, reward, terminal, info = self.env.step(action)
            acc_reward += reward
            if terminal:
                break
            previous_frame = frame
        # apenas para o SpaceInvaders é necessário pegar o máximo dos frames pois
        # o jogo pisca muito frequentemente
        if self.game_name == 'SpaceInvaders-v0' and previous_frame is not None:
            frame = np.maximum.reduce([frame, previous_frame])

        # lembrando que é necessário preprocessar o frame para que o agente receba
        # o estado ja com os 4 frames ou mais juntos
        return self.preprocess(frame), acc_reward, terminal, info
