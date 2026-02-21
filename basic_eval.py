from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig,
    EvaluationTarget,
    ApiEndpoint,
    EndpointType,
    ConfigParams
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="/data/ib-a100-cluster-a-pri-lmt_967/users/kaya/workspace/rl/results",
    params=ConfigParams(
        limit_samples=3,
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Configure target endpoint
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="Qwen/Qwen3-0.6B",
        url="http://localhost:6000/v1",
        type=EndpointType.CHAT,
        api_key="MY_API_KEY"
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")