"""ParamSeparator — splits SDK params from boto3 params."""


class ParamSeparator:
    """Separates SDK-specific params from boto3 create_transform_job params.

    SDK-specific parameters (Strategy, AllowedRegions, WorkloadDescription) are
    extracted into their own dict. Everything else — including unrecognized
    parameters — is forwarded unchanged to boto3 for forward compatibility.
    """

    SDK_PARAMS = {"Strategy", "AllowedRegions", "WorkloadDescription", "RegionConfig", "AutoCreateModel"}

    @staticmethod
    def separate(kwargs: dict) -> tuple[dict, dict, bool]:
        """Split kwargs into (boto3_params, sdk_params, is_list_mode).

        Returns:
            boto3_params: All params not in SDK_PARAMS, forwarded unchanged.
                Includes any unrecognized params — the SDK never rejects them.
            sdk_params: Extracted SDK-specific params.
            is_list_mode: True if TransformResources.InstanceType is a list.
        """
        sdk_params = {}
        boto3_params = {}
        for key, value in kwargs.items():
            if key in ParamSeparator.SDK_PARAMS:
                sdk_params[key] = value
            else:
                boto3_params[key] = value

        instance_type = (
            boto3_params.get("TransformResources", {}).get("InstanceType")
        )
        is_list_mode = isinstance(instance_type, list)

        return boto3_params, sdk_params, is_list_mode
