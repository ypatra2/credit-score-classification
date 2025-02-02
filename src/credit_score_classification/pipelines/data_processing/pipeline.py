from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_loan


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_loan,
                inputs="loan",
                outputs="preprocessed_loan",
                name="preprocess_loan_node",
            )
        ]
    )
