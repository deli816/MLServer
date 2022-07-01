import mlflow

from io import StringIO
from fastapi import Request, Response

from mlflow.exceptions import MlflowException
from mlflow.pyfunc.scoring_server import (
    CONTENT_TYPES,
    CONTENT_TYPE_CSV,
    CONTENT_TYPE_JSON,
    CONTENT_TYPE_JSON_SPLIT_ORIENTED,
    CONTENT_TYPE_JSON_RECORDS_ORIENTED,
    CONTENT_TYPE_JSON_SPLIT_NUMPY,
    parse_csv_input,
    infer_and_parse_json_input,
    parse_json_input,
    parse_split_oriented_json_input_to_numpy,
    predictions_to_json,
)
from mlserver.codecs import (
    NumpyRequestCodec,
    PandasCodec
)
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.model import MLModel
from mlserver.utils import get_model_uri
from mlserver.handlers import custom_handler
from mlserver.errors import InferenceError
from mlserver.settings import ModelParameters
from mlserver.logging import logger

from .codecs import TensorDictCodec
from .metadata import (
    to_metadata_tensors,
    to_model_content_type,
    DefaultInputPrefix,
    DefaultOutputPrefix,
)
from datetime import datetime


class MLflowRuntime(MLModel):
    """
    Implementation of the MLModel interface to load and serve `scikit-learn`
    models persisted with `joblib`.
    """

    # TODO: Decouple from REST
    @custom_handler(rest_path="/ping", rest_method="GET")
    async def ping(self, request: Request) -> str:
        """
        This custom handler is meant to mimic the behaviour of the existing
        health endpoint in MLflow's local dev server.
        For details about its implementation, please consult the original
        implementation in the MLflow repository:

            https://github.com/mlflow/mlflow/blob/master/mlflow/pyfunc/scoring_server/__init__.py
        """
        return "\n"

    # TODO: Decouple from REST
    @custom_handler(rest_path="/invocations")
    async def invocations(self, request: Request) -> Response:
        """
        This custom handler is meant to mimic the behaviour of the existing
        scoring server in MLflow.
        For details about its implementation, please consult the original
        implementation in the MLflow repository:

            https://github.com/mlflow/mlflow/blob/master/mlflow/pyfunc/scoring_server/__init__.py
        """
        content_type = request.headers.get("content-type", None)
        raw_data = await request.body()
        as_str = raw_data.decode("utf-8")

        if content_type == CONTENT_TYPE_CSV:
            print("content type csv")
            csv_input = StringIO(as_str)
            data = parse_csv_input(csv_input=csv_input)
        elif content_type == CONTENT_TYPE_JSON:
            print("content type json")
            data = infer_and_parse_json_input(as_str, self._input_schema)
        elif content_type == CONTENT_TYPE_JSON_SPLIT_ORIENTED:
            print("content type json split oriented")
            data = parse_json_input(
                json_input=StringIO(as_str),
                orient="split",
                schema=self._input_schema,
            )
        elif content_type == CONTENT_TYPE_JSON_RECORDS_ORIENTED:
            print("content type record oriented")
            data = parse_json_input(
                json_input=StringIO(as_str),
                orient="records",
                schema=self._input_schema,
            )
        elif content_type == CONTENT_TYPE_JSON_SPLIT_NUMPY:
            print("content type json split numpy")
            data = parse_split_oriented_json_input_to_numpy(as_str)
        else:
            content_type_error_message = (
                "This predictor only supports the following content types, "
                f"{CONTENT_TYPES}. Got '{content_type}'."
            )
            raise InferenceError(content_type_error_message)

        try:
            print("##### model.precition() ######, data: ", data, "#")
            raw_predictions = self._model.predict(data)
            print("##### raw prediction: ", raw_predictions)
        except MlflowException as e:
            raise InferenceError(e.message)
        except Exception:
            error_message = (
                "Encountered an unexpected error while evaluating the model. Verify"
                " that the serialized input Dataframe is compatible with the model for"
                " inference."
            )
            raise InferenceError(error_message)

        result = StringIO()
        predictions_to_json(raw_predictions, result)
        return Response(content=result.getvalue(), media_type="application/json")

    async def load(self) -> bool:
        # TODO: Log info message
        print("### mlflow load model")
        model_uri = await get_model_uri(self._settings)
        print("### model_uri: ", model_uri)
        self._model = mlflow.pyfunc.load_model(model_uri)

        self._input_schema = self._model.metadata.get_input_schema()
        self._signature = self._model.metadata.signature
        self._sync_metadata()

        self.ready = True
        return self.ready

    def _sync_metadata(self) -> None:
        print("### mlflow sync metadata")
        # Update metadata from model signature (if present)
        if self._signature is None:
            return

        if self.inputs:
            logger.warning("Overwriting existing inputs metadata with model signature")

        self.inputs = to_metadata_tensors(
            schema=self._signature.inputs, prefix=DefaultInputPrefix
        )

        if self.outputs:
            logger.warning("Overwriting existing outputs metadata with model signature")

        self.outputs = to_metadata_tensors(
            schema=self._signature.outputs, prefix=DefaultOutputPrefix
        )

        if not self._settings.parameters:
            self._settings.parameters = ModelParameters()

        if self._settings.parameters.content_type:
            logger.warning(
                "Overwriting existing request-level content type with model signature"
            )
        print("### mlflow signature.inputs:", self._signature.inputs)
        self._settings.parameters.content_type = to_model_content_type(
            schema=self._signature.inputs
        )

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        # date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        start_time = datetime.now()
        print(start_time.strftime("%m/%d/%Y, %H:%M:%S.%f"), ": ######## start predict ########")
        decoded_payload = self.decode_request(payload, default_codec=PandasCodec)
        prediction_start_time = datetime.now()
        # print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f"), ": ### payload decoded: ", decoded_payload)
        #print(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f"), ": ### payload decoded: ")
        # print("########call prediction from wrapper func########")
        model_output = self._model.predict(decoded_payload)
        finish_time = datetime.now()
        decode_latency = prediction_start_time.timestamp() * 1000 - start_time.timestamp() * 1000
        prediction_latency = finish_time.timestamp() * 1000 - prediction_start_time.timestamp() * 1000
        overall_latency = finish_time.timestamp() * 1000 - start_time.timestamp() * 1000
        print(finish_time.strftime("%m/%d/%Y, %H:%M:%S.%f"), ": prediction done, decode latency: ", decode_latency, ", prediction latency: ", prediction_latency, " worker latency: ", overall_latency )
        return self.encode_response(model_output, default_codec=TensorDictCodec)
