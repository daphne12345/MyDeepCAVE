# noqa: D400
"""
# Ablation Paths

This module evaluates the ablation paths.

Ablation Paths is a method to analyze the importance of hyperparameters in a configuration space.
Starting from a default configuration, the default configuration is iteratively changed to the
incumbent configuration by changing one hyperparameter at a time, choosing the
hyperparameter that leads to the largest improvement in the objective function at each step.

## Classes:
    - Ablation: Provide an evaluator of the ablation paths.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import copy
from collections import OrderedDict

import numpy as np

from deepcave.evaluators.epm.random_forest_surrogate import RandomForestSurrogate
from deepcave.evaluators.ablation import Ablation
from deepcave.runs import AbstractRun
from deepcave.runs.objective import Objective
from deepcave.utils.logs import get_logger
import pandas as pd

class WeightedModel:
    def __init__(self, seed, n_trees):
        self.models = [RandomForestSurrogate(self.cs, seed=seed, n_trees=n_trees), RandomForestSurrogate(self.cs, seed=seed, n_trees=n_trees)]

    def train(self, X, Ys):
        """
        Train a Random Forest model as surrogate to predict the performance of a configuration. Where the input data are the
        configurations and the objective.
        :param group: the runs as group
        :param df: dataframe containing the encoded configurations
        :param objective: the name of the target column
        :return: the trained model
        """
        for model, Y in zip(self.models,Ys):
            model.fit(X, Y)

    def predict(self, cfg, weighting):
        """
        Predicts the performance of the input configuration with the input list of models. The model results are normalized,
        weighted by the input wieghtings and summed.
        :param group: the runs as group
        :param cfg: configuration
        :param models: list of models
        :param weighting: tuple of weightings
        :return: the mean and variance of the normalized weighted sum of predictions
        """
        mean, var = 0, 0
        obj = 0
        for model, w in zip(self.models, weighting):
            all_predictions = np.stack([tree.predict(cfg) for tree in model.estimators_])
            normed_preds = (all_predictions - all_predictions.min()) / (all_predictions.max() - all_predictions.min())
            mean += w * np.mean(normed_preds, axis=0)
            var += w * np.var(normed_preds, axis=0)
            obj += 1
        return mean, var



class MOAblation(Ablation):
    """
    Provide an evaluator of the ablation paths.

    Properties
    ----------
    run : AbstractRun
        The run to analyze.
    cs : ConfigurationSpace
        The configuration space of the run.
    hp_names : List[str]
        A list of the hyperparameter names.
    performances : Optional[Dict[Any, Any]]
        A dictionary containing the performances for each HP.
    improvements : Optional[Dict[Any, Any]]
        A dictionary containing the improvements over the respective previous step for each HP.
    objectives : Optional[Union[Objective, List[Objective]]]
        The objective(s) of the run.
    default_config : Configurations
        The default configuration of this configuration space.
        Gets changed step by step towards the incumbent configuration.
    """

    def __init__(self, run: AbstractRun):
        super().__init__(run)

    def calculate(
        self,
        objectives: Optional[Union[Objective, List[Objective]]],  # noqa
        budget: Optional[Union[int, float]] = None,  # noqa
        n_trees: int = 50,  # noqa
        seed: int = 0,  # noqa
    ) -> None:
        """
        Calculate the ablation path performances and improvements.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]]
            The objective(s) to be considered.
        budget : Optional[Union[int, float]]
            The budget to be considered. If None, all budgets of the run are considered.
            Default is None.
        n_trees : int
            The number of trees for the surrogate model.
            Default is 50.
        seed : int
            The seed for the surrogate model.
            Default is 0.
        """
        for objective in objectives:
            assert isinstance(objective, Objective)

        performances: OrderedDict = OrderedDict()
        improvements: OrderedDict = OrderedDict()

        df = self.run.get_encoded_data(objectives, budget, specific=True, include_config_ids=True)

        # Obtain all configurations with theirs costs
        df = df.dropna(subset=[obj.name for obj in objectives])
        X = df[list(self.run.configspace.keys())].to_numpy()

        # normalize objectives
        objectives_normed = list()
        for obj in objectives:
            normed = obj.name + '_normed'
            df[normed] = (df[obj.name] - df[obj.name].min()) / (df[obj.name].max() - df[obj.name].min())
            objectives_normed.append(normed)


        Ys = [df[obj.name].to_numpy() for obj in objectives_normed]
        self._model = WeightedModel(seed, n_trees)
        self._model.train(X, Ys)

        df_all = pd.DataFrame([])
        weightings = self.get_weightings(objectives_normed, df)

        for w in weightings:
            for w in weightings:
                df_res, default = calculate_ablation_path(group, df, objectives_normed, w, models)
                df_res = df_res.drop(columns=['index'])
                df_res['weight_for_' + objectives_normed[0]] = w[0]
                df_all = pd.concat([df_all, df_res])




        # Get the incumbent configuration
        incumbent_config, _ = self.run.get_incumbent(budget=budget, objectives=objective)
        incumbent_encode = self.run.encode_config(incumbent_config)

        # Get the default configuration
        self.default_config = self.cs.get_default_configuration()
        default_encode = self.run.encode_config(self.default_config)

        # Obtain the predicted cost of the default and incumbent configuration
        def_cost, def_std = self._model.predict(np.array([default_encode]))
        def_cost, def_std = def_cost[0], def_std[0]
        inc_cost, _ = self._model.predict(np.array([incumbent_encode]))

        # For further calculations, assume that the objective is to be minimized
        if objective.optimize == "upper":
            def_cost = -def_cost
            inc_cost = -inc_cost

        if inc_cost > def_cost:
            self.logger.warning(
                "The predicted incumbent objective is worse than the predicted default "
                f"objective for budget: {budget}. Aborting ablation path calculation."
            )
            performances = OrderedDict({hp_name: (0, 0) for hp_name in ["default"] + self.hp_names})
            improvements = OrderedDict({hp_name: (0, 0) for hp_name in ["default"] + self.hp_names})
        else:
            # Copy the hps names as to not remove objects from the original list
            hp_it = self.hp_names.copy()

            # Add improvement and performance of the default configuration
            improvements["default"] = (0, 0)
            if objective.optimize == "upper":
                performances["default"] = (-def_cost, def_std)
            else:
                performances["default"] = (def_cost, def_std)

            for i in range(len(hp_it)):
                # Get the results of the current ablation iteration
                continue_ablation, max_hp, max_hp_cost, max_hp_std = self._ablation(
                    objective, budget, incumbent_config, def_cost, hp_it
                )

                if not continue_ablation:
                    break

                if objective.optimize == "upper":
                    # For returning the importance, flip back the objective if it was flipped before
                    performances[max_hp] = (-max_hp_cost, max_hp_std)
                else:
                    performances[max_hp] = (max_hp_cost, max_hp_std)
                impr_std = np.sqrt(def_std**2 + max_hp_std**2)
                improvements[max_hp] = ((def_cost - max_hp_cost), impr_std)
                # New 'default' cost and std
                def_cost = max_hp_cost
                def_std = max_hp_std
                # Remove the current best hp for keeping the order right
                hp_it.remove(max_hp)

        self.performances = performances
        self.improvements = improvements

    def get_ablation_performances(self) -> Optional[Dict[Any, Any]]:
        """
        Get the ablation performances.

        Returns
        -------
        Optional[Dict[Any, Any]]
            A dictionary containing the ablation performances.

        Raises
        ------
        RuntimeError
            If the ablation performances have not been calculated.
        """
        if self.performances is None:
            raise RuntimeError("Ablation performances scores must be calculated first.")
        return self.performances

    def get_ablation_improvements(self) -> Optional[Dict[Any, Any]]:
        """
        Get the ablation improvements.

        Returns
        -------
        Optional[Dict[Any, Any]]
            A dictionary containing the ablation improvements.

        Raises
        ------
        RuntimeError
            If the ablation improvements have not been calculated.
        """
        if self.improvements is None:
            raise RuntimeError("Ablation improvements must be calculated first.")

        return self.improvements

    def _ablation(
        self,
        objective: Objective,
        budget: Optional[Union[int, float]],
        incumbent_config: Any,
        def_cost: Any,
        hp_it: List[str],
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Calculate the ablation importance for each hyperparameter.

        Parameters
        ----------
        objective: Objective
            The objective to be considered.
        budget: Optional[Union[int, float]]
            The budget of the run.
        incumbent_config: Any
            The incumbent configuration.
        def_cost: Any
            The default cost.
        hp_it: List[str]
            A list of the HPs that still have to be looked at.

        Returns
        -------
        Tuple[Any, Any, Any, Any]
            continue_ablation, max_hp, max_hp_performance, max_hp_std
        """
        max_hp = ""
        max_hp_difference = -np.inf

        for hp in hp_it:
            if hp in incumbent_config.keys() and hp in self.default_config.keys():
                config_copy = copy.copy(self.default_config)
                config_copy[hp] = incumbent_config[hp]

                new_cost, _ = self._model.predict(np.array([self.run.encode_config(config_copy)]))
                if objective.optimize == "upper":
                    new_cost = -new_cost

                difference = def_cost - new_cost

                # Check for the maximum difference hyperparameter in this round
                if difference >= max_hp_difference:
                    max_hp = hp
                    max_hp_difference = difference
            else:
                continue
        hp_count = len(list(self.cs.keys()))
        if max_hp != "":
            # For the maximum impact hyperparameter, switch the default with the incumbent value
            self.default_config[max_hp] = incumbent_config[max_hp]
            max_hp_cost, max_hp_std = self._model.predict(
                np.array([self.run.encode_config(self.default_config)])
            )
            if objective.optimize == "upper":
                max_hp_cost = -max_hp_cost
            return True, max_hp, max_hp_cost[0], max_hp_std[0]
        else:
            self.logger.info(
                f"End ablation at step {hp_count - len(hp_it) + 1}/{hp_count} "
                f"for budget {budget} (remaining hyperparameters not activate in incumbent or "
                "default configuration)."
            )
            return False, None, None, None
