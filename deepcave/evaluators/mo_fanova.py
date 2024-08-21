# TODO
#  noqa: D400
"""
# fANOVA

This module provides a tool for assessing the importance of an algorithms Hyperparameters.

Utilities provide calculation of the data wrt the budget and train the forest on the encoded data.

## Classes
    - fANOVA: Calculate and provide midpoints and sizes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import itertools as it

import numpy as np

from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.evaluators import fANOVA
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.logs import get_logger
import pandas as pd


class MOfANOVA:
    """
    Calculate and provide midpoints and sizes.

    They are generated from the forest's split values in order to get the marginals.

    Properties
    ----------
    run : AbstractRun
        The Abstract Run used for the calculation.
    cs : ConfigurationSpace
        The configuration space of the run.
    hps : List[Hyperparameters]
        The Hyperparameters of the configuration space.
    hp_names : List[str]
        The corresponding names of the Hyperparameters.
    n_trees : int
        The number of trees.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace
        self.hps = self.cs.get_hyperparameters()
        self.hp_names = self.cs.get_hyperparameter_names()
        self.logger = get_logger(self.__class__.__name__)

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        n_trees: int = 16,
        seed: int = 0,
    ) -> None:
        """
        Get the data with respect to budget and train the forest on the encoded data.

        Note
        ----
        Right now, only `n_trees` is used. It can be further specified if needed.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget is chosen.
        n_trees : int, optional
            How many trees should be used. By default 16.
        seed : int
            Random seed. By default 0.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        self.n_trees = n_trees

        # Get data
        df = self.run.get_encoded_data(
            objectives, budget, specific=True, include_combined_cost=True
        )
        X = df[self.hp_names].to_numpy()
        # Combined cost name includes the cost of all selected objectives
        Y = df[COMBINED_COST_NAME].to_numpy()

        # Get model and train it
        self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
        self._model.train(X, Y)

    def get_importances(
        self, hp_names: Optional[List[str]] = None, depth: int = 1, sort: bool = True
    ) -> Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]:
        """
        Return the importance scores from the passed Hyperparameter names.

        Warning
        -------
        Using a depth higher than 1 might take much longer.

        Parameters
        ----------
        hp_names : Optional[List[str]]
            Selected Hyperparameter names to get the importance scores from. If None, all
            Hyperparameters of the configuration space are used.
        depth : int, optional
            How often dimensions should be combined. By default 1.
        sort : bool, optional
            Whether the Hyperparameters should be sorted by importance. By default True.

        Returns
        -------
        Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]
            Dictionary with Hyperparameter names and the corresponding importance scores.
            The values are tuples of the form (mean individual, var individual, mean total,
            var total). Note that individual and total are the same if depth is 1.

        Raises
        ------
        RuntimeError
            If there is zero total variance in all trees.
        """
        if hp_names is None:
            hp_names = self.cs.get_hyperparameter_names()

        hp_ids = []
        for hp_name in hp_names:
            hp_ids.append(self.cs.get_idx_by_hyperparameter_name(hp_name))

        # Calculate the marginals
        vu_individual, vu_total = self._model.compute_marginals(hp_ids, depth)

        importances: Dict[Tuple[Any, ...], Tuple[float, float, float, float]] = {}
        for k in range(1, len(hp_ids) + 1):
            if k > depth:
                break

            for sub_hp_ids in it.combinations(hp_ids, k):
                sub_hp_ids = tuple(sub_hp_ids)

                # clean here to catch zero variance in a trees
                non_zero_idx = np.nonzero(
                    [self._model.trees_total_variance[t] for t in range(self.n_trees)]
                )

                if len(non_zero_idx[0]) == 0:
                    self.logger.warning("Encountered zero total variance in all trees.")
                    importances[sub_hp_ids] = (
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    )
                    continue

                fractions_total = np.array(
                    [
                        vu_total[sub_hp_ids][t] / self._model.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )
                fractions_individual = np.array(
                    [
                        vu_individual[sub_hp_ids][t] / self._model.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )

                importances[sub_hp_ids] = (
                    np.mean(fractions_individual),
                    np.var(fractions_individual),
                    np.mean(fractions_total),
                    np.var(fractions_total),
                )

        # Sort by total mean fraction
        if sort:
            importances = {
                k: v for k, v in sorted(importances.items(), key=lambda item: item[1][2])
            }

        # The ids get replaced with hyperparameter names again
        all_hp_names = self.cs.get_hyperparameter_names()
        importances_: Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]] = {}
        for hp_ids_importances, values in importances.items():
            hp_names = [all_hp_names[hp_id] for hp_id in hp_ids_importances]
            hp_names_key: Union[Tuple[str, ...], str]
            if len(hp_names) == 1:
                hp_names_key = hp_names[0]
            else:
                hp_names_key = tuple(hp_names)
            importances_[hp_names_key] = values

        return importances_





class fANOVAWeighted(fANOVA):
    """
    Calculate and provide midpoints and sizes from the forest's split values in order to get the marginals.
    Overriden to train the random forest with an arbitrary weighting of the objectives.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        super().__init__(run)
        self.n_trees = 100

    def get_weightings(self, objectives_normed, df):
        """
        Returns the weighting used for the weighted importance. It uses the points on the pareto-front as weightings
        :param objectives_normed: the normalized objective names as a list of strings
        :param df: dataframe containing the encoded data
        :return: the weightings as a list of lists
        """
        optimized = self.is_pareto_efficient(df[objectives_normed].to_numpy())
        return df[optimized][objectives_normed].T.apply(lambda values: values / values.sum()).T.to_numpy()

    def is_pareto_efficient(serlf, costs):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i] > c, axis=1)) and np.all(np.any(costs[i + 1:] > c, axis=1))
        return is_efficient


    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]] = None,
        budget: Optional[Union[int, float]] = None,
        n_trees: int = 16,
        seed: int = 0,
    ) -> None:
        """
        Get the data with respect to budget and train the forest on the encoded data.
        # TODO fix Kommentar
        Calculates weighted fAnova for multiple objectives.
        :param group: the runs as group
        :param df: dataframe containing the encoded data
        :param objectives_normed: the normalized objective names as a list of strings
        :param weightings: the weightings as list of lists
        :return: the dataframe containing the importances for each hyperparameter per weighting


        Note
        ----
        Right now, only `n_trees` is used. It can be further specified if needed.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default None. If None, all objectives are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget is chosen.
        n_trees : int, optional
            How many trees should be used. By default 16.
        seed : int
            Random seed. By default 0.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        self.n_trees = n_trees

        # Get data
        df = self.run.get_encoded_data(
            objectives, budget, specific=True, include_combined_cost=True
        )
        X = df[self.hp_names].to_numpy()

        # normalize objectives
        objectives_normed = list()
        for obj in objectives:
            normed = obj.name + '_normed'
            df[normed] = (df[obj.name] - df[obj.name].min()) / (df[obj.name].max() - df[obj.name].min())
            objectives_normed.append(normed)

        # TODO return df_all somwhow
        # TODO create MO plot instead of single objective
        df_all = pd.DataFrame([])
        weightings = self.get_weightings(objectives_normed, df)
        for w in weightings:
            Y = sum(df[obj] * weighting for obj, weighting in zip(objectives_normed, w)).to_numpy()

            self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
            self._model.train(X, Y)
            df_res = pd.DataFrame(self._model.get_importances(hp_names=None)).loc[0:1].T.reset_index()
            df_res['weight_for_' + objectives_normed[0]] = w[0]
            df_all = pd.concat([df_all, df_res])
        df_all = df_all.rename(columns={0: 'fanova', 1: 'variance', 'index': 'hp_name'})

