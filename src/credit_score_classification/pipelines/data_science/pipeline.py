from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, scale_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_loan", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=scale_data,
                inputs=["X_train", "X_test"],
                outputs=["X_train_scaled", "X_test_scaled"],
                name="scale_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_scaled", "y_train", "params:model_options"],
                outputs="rf_classifier",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["rf_classifier", "X_test_scaled", "y_test"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
