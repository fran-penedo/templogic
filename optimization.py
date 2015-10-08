'''
differential_evolution: The differential evolution global optimization algorithm

Copied and modified source from SciPy: scipy/optimize/_differentialevolution.py

Created on Oct 8, 2015

@author: Cristian
'''

from __future__ import division, print_function, absolute_import
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
import numbers

_MACHEPS = np.finfo(np.float64).eps


def constrained_differential_evolution(func, bounds,
                           custom_constraint=lambda x: x,
                           args=(),
                           strategy='best1bin',
                           maxiter=None, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube'):
    """See scipy.optimize.differential_evolution() function."""

    solver = ConstrainedDifferentialEvolutionSolver(func, bounds,
                                         custom_constraint=custom_constraint,
                                         args=args,
                                         strategy=strategy, maxiter=maxiter,
                                         popsize=popsize, tol=tol,
                                         mutation=mutation,
                                         recombination=recombination,
                                         seed=seed, polish=polish,
                                         callback=callback,
                                         disp=disp,
                                         init=init)
    return solver.solve()


class ConstrainedDifferentialEvolutionSolver(object):
    """See scipy.optimize.DifferentialEvolutionSolver class."""

    # Dispatch of mutation strategy method (binomial or exponential).
    _binomial = {'best1bin': '_best1',
                 'randtobest1bin': '_randtobest1',
                 'best2bin': '_best2',
                 'rand2bin': '_rand2',
                 'rand1bin': '_rand1'}
    _exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    def __init__(self, func, bounds, custom_constraint=lambda x: x, args=(),
                 strategy='best1bin', maxiter=None, popsize=15,
                 tol=0.01, mutation=(0.5, 1), recombination=0.7, seed=None,
                 maxfun=None, callback=None, disp=False, polish=True,
                 init='latinhypercube'):

        if strategy in self._binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self._exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            raise ValueError("Please select a valid mutation strategy")
        self.strategy = strategy

        self.callback = callback
        self.polish = polish
        self.tol = tol

        #Mutation constant should be in [0, 2). If specified as a sequence
        #then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        self._custom_constraint = custom_constraint

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2
                or not np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        self.maxiter = maxiter or 1000
        self.maxfun = (maxfun or ((self.maxiter + 1) * popsize *
                                  np.size(self.limits, 1)))

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        parameter_count = np.size(self.limits, 1)
        self.random_number_generator = _make_random_gen(seed)

        #default initialization is a latin hypercube design, but there
        #are other population initializations possible.
        self.population = np.zeros((popsize * parameter_count,
                                    parameter_count))
        if init == 'latinhypercube':
            self.init_population_lhs()
        elif init == 'random':
            self.init_population_random()
        else:
            raise ValueError("The population initialization method must be one"
                             "of 'latinhypercube' or 'random'")

        self.population_energies = np.ones(
            popsize * parameter_count) * np.inf

        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling
        Latin Hypercube Sampling ensures that the sampling of parameter space
        is maximised.
        """
        samples = np.size(self.population, 0)
        N = np.size(self.population, 1)
        rng = self.random_number_generator

        # Generate the intervals
        segsize = 1.0 / samples

        # Fill points uniformly in each interval
        rdrange = rng.rand(samples, N) * segsize
        rdrange += np.atleast_2d(
            np.linspace(0., 1., samples, endpoint=False)).T

        # Make the random pairings
        self.population = np.zeros_like(rdrange)

        for j in range(N):
            order = rng.permutation(range(samples))
            self.population[:, j] = rdrange[order, j]

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population.shape)
        for p in self.population:
            self._ensure_constraint(p)

    @property
    def x(self):
        """
        The best solution from the solver

        Returns
        -------
        x - ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """

        nfev, nit, warning_flag = 0, 0, False
        status_message = _status_message['success']

        # calculate energies to start with
        for index, candidate in enumerate(self.population):
            parameters = self._scale_parameters(candidate)
            self.population_energies[index] = self.func(parameters,
                                                        *self.args)
            nfev += 1

            if nfev > self.maxfun:
                warning_flag = True
                status_message = _status_message['maxfev']
                break

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

        if warning_flag:
            return OptimizeResult(
                           x=self.x,
                           fun=self.population_energies[0],
                           nfev=nfev,
                           nit=nit,
                           message=status_message,
                           success=(warning_flag is not True))

        # do the optimisation.
        for nit in range(1, self.maxiter + 1):
            if self.dither is not None:
                self.scale = self.random_number_generator.rand(
                ) * (self.dither[1] - self.dither[0]) + self.dither[0]
            for candidate in range(np.size(self.population, 0)):
                if nfev > self.maxfun:
                    warning_flag = True
                    status_message = _status_message['maxfev']
                    break

                trial = self._mutate(candidate)
                self._ensure_constraint(trial)
                parameters = self._scale_parameters(trial)

                energy = self.func(parameters, *self.args)
                nfev += 1

                if energy < self.population_energies[candidate]:
                    self.population[candidate] = trial
                    self.population_energies[candidate] = energy

                    if energy < self.population_energies[0]:
                        self.population_energies[0] = energy
                        self.population[0] = trial

            # stop when the fractional s.d. of the population is less than tol
            # of the mean energy
            convergence = (np.std(self.population_energies) /
                           np.abs(np.mean(self.population_energies) +
                                  _MACHEPS))

            if self.disp:
                print("differential_evolution step %d: f(x)= %g"
                      % (nit,
                         self.population_energies[0]))

            if (self.callback and
                    self.callback(self._scale_parameters(self.population[0]),
                                  convergence=self.tol / convergence) is True):

                warning_flag = True
                status_message = ('callback function requested stop early '
                                  'by returning True')
                break

            if convergence < self.tol or warning_flag:
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)

            nfev += result.nfev
            DE_result.nfev = nfev

            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return DE_result

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index, param in enumerate(trial):
            if param > 1 or param < 0:
                trial[index] = self.random_number_generator.rand()
        # ADDED: imposed custom constraints
        trial[:] = self._custom_constraint(trial)

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])
        parameter_count = np.size(trial, 0)

        fill_point = self.random_number_generator.randint(0, parameter_count)

        if (self.strategy == 'randtobest1exp'
                or self.strategy == 'randtobest1bin'):
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = self.random_number_generator.rand(parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < parameter_count and
                   self.random_number_generator.rand() <
                   self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _randtobest1(self, candidate, samples):
        """
        randtobest1bin, randtobest1exp
        """
        r0, r1 = samples[:2]
        bprime = np.copy(self.population[candidate])
        bprime += self.scale * (self.population[0] - bprime)
        bprime += self.scale * (self.population[r0] -
                                self.population[r1])
        return bprime

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                            (self.population[r0] + self.population[r1]
                           - self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                 (self.population[r1] + self.population[r2] -
                  self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(np.size(self.population, 0)),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(np.size(self.population, 0)))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs


def _make_random_gen(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

if __name__ == '__main__':
    pass
