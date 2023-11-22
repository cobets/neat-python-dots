from neat import ParallelEvaluator


class DotsParallelEvaluator(ParallelEvaluator):
    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, genomes, config)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)
