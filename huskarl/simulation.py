from itertools import count
from collections import namedtuple
from queue import Empty
from time import sleep
import multiprocessing as mp

import numpy as np
import cloudpickle  # For pickling lambda functions and more

from huskarl.memory import Transition
from huskarl.core import HkException

# Packet used to transmit experience from environment subprocesses to main process
# The first packet of every episode will have reward and info set to None
# The last packet of every episode will have done set to True
Experience = namedtuple('Experience', ['reward', 'state', 'done', 'info'])


def print_episode(instance: int, observation: object, reward: float,
                  info: dict):
    '''Logs the information of a finished episode of an instance to the console.
    Replace with other function to log to memory or file.

    Parameters
    ----------
    instance: int
        Id of the instance that has finished an episode.
    observation: object
        Final observation of the episode.
    reward: float
        Total reward of the episode.
    info:
    '''
    print(f'instance: {instance}\nreward: {reward}\ninfo: {info}')


def print_rewards(episode_rewards: list, episode_steps: list, done=False):
    '''Outputs the rewards to the console. Replace with your own plot function.

    Parameters
    ----------
    episode_rewards: list
        Final rewards of episode of all instances.
    episode_steps: list
        Step count for each of the final episode rewards.
    done: bool
        If the simulation is finished.'''
    for i in range(len(episode_steps)):
        # or matplotlib.pyplot.plot(episode_steps[i], episode_rewards[i])
        result = [episode_steps[i], episode_rewards[i]]
        result = list(result)
        print(f'Instance {i} (step, reward): {result}')
    print(f'Done = {done}')


class Simulation:
    """Simulates an agent interacting with one of multiple environments."""

    def __init__(self, create_env, agent, mapping=None):
        self.create_env = create_env
        self.agent = agent
        self.mapping = mapping

    def train(self, max_steps=100_000, instances=1, visualize=False,
              plot=print_rewards, max_subprocesses=0,
              log_episode=print_episode):
        """Trains the agent on the specified number of environment instances.

        Parameters
        ----------
        max_steps: int
            Number of steps to execute in one training epoch.
        instances: int
            Number of parallel executing policies.
        visualize: bool
            Do call the render method of the environment
        plot: function
            Plot the rewards after every finished episode (of any instance).
            If None no plot will be generated.
        log_episode: function
            Callback to log the infos of an episode.
            If None nothing will be logged."""

        self.agent.training = True
        if max_subprocesses == 0:
            # Use single process implementation
            self._sp_train(max_steps, instances, visualize, plot, log_episode)
        elif max_subprocesses is None or max_subprocesses > 0:
            # Use multiprocess implementation
            self._mp_train(max_steps, instances, visualize,
                           plot, max_subprocesses, log_episode)
        else:
            raise HkException(
                f"Invalid max_subprocesses setting: {max_subprocesses}")

    def _sp_train(self, max_steps, instances, visualize, plot, log_episode):
        """Trains using a single process."""
        # Keep track of rewards per episode per instance
        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        # Create and initialize environment instances
        envs = [self.create_env() for i in range(instances)]
        states = [env.reset() for env in envs]

        for step in range(max_steps):
            for i in range(instances):
                if visualize:
                    envs[i].render()
                action = self.agent.act(states[i], i)
                next_state, reward, done, info = envs[i].step(action)
                self.agent.push(Transition(
                    states[i], action, reward, None if done else next_state),
                    i)
                episode_rewards[i] += reward
                if done:
                    episode_reward_sequences[i].append(episode_rewards[i])
                    episode_step_sequences[i].append(step)
                    if plot:
                        plot(episode_reward_sequences, episode_step_sequences)
                    if log_episode:
                        log_episode(i, states[i], episode_rewards[i], info)
                    episode_rewards[i] = 0
                    states[i] = envs[i].reset()
                else:
                    states[i] = next_state
            # Perform one step of the optimization
            self.agent.train(step)

        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def _mp_train(self, max_steps, instances, visualize, plot,
                  max_subprocesses, log_episode):
        """Trains using multiple processes.
        Useful to parallelize the computation of heavy environments.

        Parameters
        ----------
        log_info(info: dict)
            Callback to log the infos of an episode"""
        # Unless specified set the maximum number of processes to be the number
        # of cores in the machine
        if max_subprocesses is None:
            max_subprocesses = mp.cpu_count()
        n_processes = min(instances, max_subprocesses)

        # Split instances into processes as homogeneously as possibly
        instances_per_process = [instances//n_processes] * n_processes
        leftover = instances % n_processes
        if leftover > 0:
            for i in range(leftover):
                instances_per_process[i] += 1

        # Create a unique id (index) for each instance, grouped by process
        instance_ids = [list(range(i, instances, n_processes))[:ipp]
                        for i, ipp in enumerate(instances_per_process)]

        # Create processes and pipes (one pipe for each environment instance)
        pipes = []
        processes = []
        for i in range(n_processes):
            child_pipes = []
            for j in range(instances_per_process[i]):
                parent, child = mp.Pipe()
                pipes.append(parent)
                child_pipes.append(child)
            p_args = (cloudpickle.dumps(self.create_env),
                      instance_ids[i], max_steps, child_pipes, visualize)
            processes.append(mp.Process(target=_train, args=p_args))

        # Start all processes
        print((f'Starting {n_processes} process(es) for '
               f'{instances} environment instance(s)... {instance_ids}'))
        for p in processes:
            p.start()

        # Keep track of rewards per episode per instance
        episode_reward_sequences = [[] for i in range(instances)]
        episode_step_sequences = [[] for i in range(instances)]
        episode_rewards = [0] * instances

        # Temporarily record Experience instances received from each subprocess
        # Each Transition instance requires two Experience instances to be
        # created
        exp_buffer = [None] * instances

        # Keep track of last actions sent to subprocesses
        last_actions = [None] * instances

        for step in range(max_steps):

            # Keep track from which environments we have already constructed a
            # full Transition instance
            # and sent it to agent. This is to synchronize steps.
            step_done = [False] * instances

            # Steps across environments are synchronized
            while sum(step_done) < instances:

                # Within each step, Transitions are received and processed on a
                # first-come first-served basis
                awaiting_pipes = [p for iid, p in enumerate(
                    pipes) if step_done[iid] == 0]
                ready_pipes = mp.connection.wait(awaiting_pipes, timeout=None)
                pipe_indexes = [pipes.index(rp) for rp in ready_pipes]

                # Do a round-robin over processes to best divide computation
                pipe_indexes.sort()
                for iid in pipe_indexes:
                    # Receive a Experience
                    experience = pipes[iid].recv()

                    # If we already had a Experience for this environment then
                    # we are able to create and push a Transition
                    if exp_buffer[iid] is not None:
                        transition = Transition(
                            exp_buffer[iid].state, last_actions[iid],
                            experience.reward, experience.state)
                        self.agent.push(transition, iid)
                        step_done[iid] = True
                    exp_buffer[iid] = experience

                    # Add reward
                    if experience.reward:
                        episode_rewards[iid] += experience.reward
                    # Check if episode is done
                    if experience.done:
                        # Episode is done - store rewards, log and update plot
                        episode_reward_sequences[iid].append(
                            episode_rewards[iid])
                        episode_step_sequences[iid].append(step)
                        if log_episode:
                            log_episode(iid, experience.state,
                                        experience.reward, experience.info)
                        if plot:
                            plot(episode_reward_sequences,
                                 episode_step_sequences)
                        exp_buffer[iid] = None
                        episode_rewards[iid] = 0
                    else:
                        # Episode is NOT done - act according to state and send
                        # action to the subprocess
                        action = self.agent.act(experience.state, iid)
                        last_actions[iid] = action
                        try:
                            pipes[iid].send(action)
                        # Disregard BrokenPipeError on last step
                        except BrokenPipeError as bpe:
                            if step < (max_steps - 1):
                                raise bpe

            # Train the agent at the end of every synchronized step
            self.agent.train(step)

        if plot:
            plot(episode_reward_sequences, episode_step_sequences, done=True)

    def test(self, max_steps: int, visualize=True, log_episode=print_episode):
        """Test the agent on the environment.

        Parameters
        ----------
        max_steps: int
            Number of steps to execute in one training epoch.
        visualize: bool
            Do call the render method of the environment
        log_episode: function
            Callback to log the infos of an episode.
            If None nothing will be logged."""
        self.agent.training = False

        # Create and initialize environment
        env = self.create_env()
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if visualize:
                env.render()
            action = self.agent.act(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            if done:
                if log_episode:
                    log_episode(0, state, episode_reward, info)
                episode_reward = 0
                state = env.reset()


def _train(create_env, instance_ids, max_steps, pipes, visualize):
    """This function is to be executed in a subprocess."""
    pipes = {iid: p for iid, p in zip(instance_ids, pipes)}
    # Reused dictionary of actions
    actions = {iid: None for iid in instance_ids}

    # Initialize environments and send initial state to agent in parent process
    create_env = cloudpickle.loads(create_env)
    envs = {iid: create_env() for iid in instance_ids}
    for iid in instance_ids:
        state = envs[iid].reset()
        pipes[iid].send(Experience(0, state, False, None))

    # Run for the specified number of steps
    for step in range(max_steps):
        for iid in instance_ids:
            # Get action from agent in main process via pipe
            actions[iid] = pipes[iid].recv()
            if visualize:
                envs[iid].render()

            # step environment and send experience
            state, reward, done, info = envs[iid].step(actions[iid])
            pipes[iid].send(Experience(reward, state, done, info))

            if done:
                # reset environment
                state = envs[iid].reset()
                # send initial state
                pipes[iid].send(Experience(0, state, False, None))
