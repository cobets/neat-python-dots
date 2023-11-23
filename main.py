import os
import argparse
import multiprocessing
import neat
from neat import ParallelEvaluator
from dotsenv import DotsEnv


class DotsParallelEvaluator(ParallelEvaluator):
    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, genomes, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


def eval_genome(genome, genomes, config):
    l_result = 0.0
    net1 = neat.nn.FeedForwardNetwork.create(genome, config)

    for genome2_id, genome2 in genomes:
        if genome2_id != genome.key:
            net2 = neat.nn.FeedForwardNetwork.create(genome2, config)

            state = DotsEnv(8, 8)
            turn = True
            while not state.terminal():
                if turn:
                    p = net1.activate(state.board.ravel())
                else:
                    p = net2.activate(state.board.ravel())

                action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x: -x[1])[0][0]
                state.play(action)

                turn = not turn

            r = state.terminal_reward()

            if r > 0:
                l_result += r

    return l_result


def train(p):
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    pe = DotsParallelEvaluator(multiprocessing.cpu_count() - 1, eval_genome)
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


def play(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = DotsEnv(8, 8)
    while not state.terminal():
        p = net.activate(state.board.ravel())
        action = sorted([(a, p[a]) for a in state.legal_actions()], key=lambda x: -x[1])[0][0]
        state.play(action)
        print(f'Neat: {action // 8}, {action % 8}')
        v = tuple(input("You Turn:"))
        action = int(v[0]) * 8 + int(v[1])
        state.play(action)


def build_config():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-dots.cfg')

    # Load configuration
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )


if __name__ == '__main__':
    _DESCRIPTION = """ ! """
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        '--checkpoint',
        default=None,
        help='Checkpoint file to start with. Default None',
        type=str
    )
    parser.add_argument(
        '--mode',
        default='train',
        help='Run mode: "train", "play". Default "train"',
        type=str
    )
    parser.add_argument(
        '--best_genome_id',
        default=None,
        help='Best genome id for play mode. Default first genome in population',
        type=int
    )

    args = parser.parse_args()

    p = None
    if args.checkpoint is not None:
        p = neat.Checkpointer.restore_checkpoint(args.checkpoint)
    else:
        # Create the population, which is the top-level object for a NEAT run.
        p = neat.Population(build_config())

    if args.mode == 'train':
        train(p)
    elif args.mode == 'play':
        if args.best_genome_id is None:
            best_genome_id = next(iter(p.population))
        else:
            best_genome_id = args.best_genome_id

        play(p.population[best_genome_id], p.config)
