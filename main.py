import os
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


def main():
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-dots.cfg')

    # Load configuration
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 300 generations.
    pe = DotsParallelEvaluator(multiprocessing.cpu_count() - 1, eval_genome)
    winner = p.run(pe.evaluate, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    main()
