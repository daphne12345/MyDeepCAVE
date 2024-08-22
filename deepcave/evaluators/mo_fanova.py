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
from deepcave.evaluators.fanova import fANOVA
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.logs import get_logger
import pandas as pd

class MOfANOVA(fANOVA):
    """
    Calculate and provide midpoints and sizes from the forest's split values in order to get the marginals.
    Overriden to train the random forest with an arbitrary weighting of the objectives.
    """

    def __init__(self, run: AbstractRun):
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        super().__init__(run)
        self.n_trees = 100
        self.importances_ = None

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

        df_all = pd.DataFrame([])
        weightings = self.get_weightings(objectives_normed, df)
        for w in weightings:
            Y = sum(df[obj] * weighting for obj, weighting in zip(objectives_normed, w)).to_numpy()

            self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
            self._model.train(X, Y)
            df_res = pd.DataFrame(super(MOfANOVA, self).get_importances(hp_names=None)).loc[0:1].T.reset_index()
            df_res['weight_for_' + objectives_normed[0]] = w[0]
            df_all = pd.concat([df_all, df_res])
        self.importances_ = df_all.rename(columns={0: 'importance', 1: 'variance', 'index': 'hp_name'})
        # importances_ = df_all.sort_values(by='weight_for_1-accuracy_normed').groupby('hp_name').agg(list).T.to_dict('list') # Dict[hp_name: [importances->list, variances->list, weightings->list]]
        # return importances_


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
        if hp_names:
            return self.importances_[self.importances_['hp_name'].isin(hp_names)]
        else:
            return self.importances_

