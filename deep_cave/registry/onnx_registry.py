import os
from typing import Dict, List

import onnx
import onnxruntime as rt

from ..util.logs import get_logger
from .abstract_registry import AbstractRegistry
from .onnx_surrogate import ONNXSurrogate
from .onnx_surrogate import ONNXSurrogate

logger = get_logger(__name__)


class ONNXRegistry(AbstractRegistry):
    """
    Based on the save_location a model is saved under the model_id for later retrieval.

    Missing. Extension for saving and retrieving with different protocols. As it was done in Store.
    """

    @property
    def format(self)-> str:
        """
        Maybe useful later, when there is more than one serialization format for models. Currently, ONNX is the
        only one.

        Returns
        -------
            The format used for saving. Always onnx.
        """
        return 'onnx'

    def __init__(self, save_location: str):
        """
        Initialize registry for a specific directory

        Parameters
        ----------
        save_location
            str. Describing the directory location.
        """
        super().__init__(save_location)
        # place it here because the save_location of AbstractRegistry could also be an url.
        # This subclass saves only files. So place the file checks here.
        if not os.path.isdir(self.save_location):
            raise NotADirectoryError(self.save_location + ' is not a valid directory')

    def log_surrogate_model(self, model: onnx.ModelProto, model_id: str) -> None:
        """
        Implements how to save the model.
        Expects the user to generate the onnx.ModelProto from his ML model.

        Parameters
        ----------
        model
            onnx.ModelProto. Already transformed surrogate model.
        model_id
            str. Unique name for the model generated by the system for later reference to the meta data.
        Returns
        -------
            Nothing.
        """
        model_path = os.path.join(self.save_location, model_id + '.' + self.format)
        if os.path.exists(model_path):
            logger.warning(f'File {model_path} already exists. Overriding it')
        with open(model_path, 'wb') as f:
            f.write(model.SerializeToString())

    def get_surrogate(self, model_id: str, mapping: Dict[str, List[str]]) -> ONNXSurrogate:
        """
        Retieve the serialized model, deserialize it and wrap it in an ONNXSurrogate model.
        The mapping is considered meta data and mandaged with the Store class.

        Parameters
        ----------
        model_id
            str. Unique id, by which the model is referenced.
        mapping
            Dict. The mapping between features and model input.
        Returns
        -------
            Returns an ONNXSurrogate model, which is sklearn compatible.
        """
        model_path = os.path.join(self.save_location, model_id + '.' + self.format)
        sess = rt.InferenceSession(model_path)

        return ONNXSurrogate(sess, mapping=mapping)
