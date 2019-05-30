# Comp. Engr. Bruno Brandão Soares Martins

import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, session, action_size, optimizer=tf.train.AdamOptimizer(1e-4)):
        # session: a sessão do tensorflow
        # action_size: número de ações
        self.action_size = action_size
        self.optimizer = optimizer
        self.sess = session

        with tf.variable_scope('network'):
            # guardando estado, politica e valor para a rede
            self.state, self.policy, self.value = self.build_model(84, 84, 4)

            # pegando os pesos da rede para computar os gradientes em relação a eles
            self.weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='network')

            # placeholders de ações valores alvo e vantagens, para calcular a perda
            self.action = tf.placeholder('int32', [None], name='action')
            self.target_value = tf.placeholder('float32', [None], name='target_value')
            self.advantages = tf.placeholder('float32', [None], name='advantages')

        with tf.variable_scope('optimizer'):
            # computar os vetores "one_hot" das ações para que, ao multiplica-las,
            # apenas as ações tomadas tenham suas probabilidades representadas na perda
            action_one_hot = tf.one_hot(self.action, self.action_size, 1.0, 0.0)

            # cortando as probabilidades para evitar zeros e ums
            min_prob = 0.0000000001
            max_prob = 0.9999999999
            # cálculo do log de acordo com a fórmula do gradiente
            self.log_policy = tf.log(tf.clip_by_value(self.policy, min_prob, max_prob))

            # multiplicando pela matriz "one_hot" é possível pegar apenas o valor da ação tomada
            self.log_pi_for_action = tf.reduce_sum(tf.multiply(self.log_policy, action_one_hot), reduction_indices=1)

            # advantages são calculadas fora pois a função de valor V(s) deve ser independente para
            # a política do gradiente. Desta forma as advantages são, as recompensas futuras, menos,
            # a função de valor, o valor médio daquele estado (R - V(s))
            # Seguindo a formula as advantages são multiplicadas pelo log da probabilidade da ação
            # efetivamente tomada naquele estado. O negativo existe pois o tensorflow busca diminuir
            self.policy_loss = -tf.reduce_mean(self.log_pi_for_action * self.advantages)

            # a perda da função de valor é dada similar ao erro de um classificador, MSE
            self.value_loss = tf.reduce_mean(tf.square(self.target_value - self.value))

            # a entropia deve ser uma função que aumenta quando a politica apresenta valores
            # muito similares para todas as ações. Sendo assim, ela é a soma das multiplicações
            # das probabilidades com menos o log destas. O negativo se apresenta pois os logs de
            # valores menores que 1 são negativos.
            self.entropy = tf.reduce_sum(tf.multiply(self.policy, -self.log_policy))

            # esta é a função de perda total da rede, lembrando que a tomada de decisões
            # é mais importante que avaliá-las, por isso, a ponderação.
            self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

            grads = tf.gradients(self.loss, self.weights)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)
            grads_vars = list(zip(grads, self.weights))

            self.train_op = optimizer.apply_gradients(grads_vars)

    def get_value(self, state):
        return self.sess.run(self.value, {self.state: state}).flatten()

    def get_policy(self, state):
        return self.sess.run(self.policy, {self.state: state}).flatten()

    def get_policy_and_value(self, state):
        policy, value = self.sess.run([self.policy, self.value], {self.state: state})
        return policy.flatten(), value.flatten()

    def train(self, states, actions, target_values, advantages):
        self.sess.run(self.train_op, feed_dict={
        self.state: states,
        self.action: actions,
        self.target_value: target_values,
        self.advantages: advantages })


    def build_model(self, h, w, channels):
        # placeholder da entrada, varios frames empilhados juntos
        state = tf.placeholder('float32', shape=(None, h, w, channels), name='state')

        with tf.variable_scope('conv1'):
            conv1 = tf.contrib.layers.convolution2d(inputs=state,
            num_outputs=16, kernel_size=[8,8], stride=[4,4], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer() )

        with tf.variable_scope('conv2'):
            conv2 = tf.contrib.layers.convolution2d( inputs=conv1, num_outputs=32,
            kernel_size=[4,4], stride=[2,2], padding="VALID",
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            biases_initializer=tf.zeros_initializer() )

        with tf.variable_scope('flatten'):
            flatten = tf.contrib.layers.flatten(inputs=conv2)

        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(inputs=flatten, num_outputs=256,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer() )

        # saída da política/ator
        with tf.variable_scope('policy'):
            policy = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=self.action_size,
            activation_fn=tf.nn.softmax,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None )

        # saída do valor/crítico
        with tf.variable_scope('value'):
            value = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1,
            activation_fn=None,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=None )

        return state, policy, value
