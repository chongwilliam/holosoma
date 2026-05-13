from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig
from holosoma.config_types.video import VideoConfig

disabled = DisabledLoggerConfig()

wandb = WandbLoggerConfig(mode="online")

wandb_offline = WandbLoggerConfig(mode="offline")

debug_video = DisabledLoggerConfig(
    video=VideoConfig(
        enabled=True,
        interval=1,
        upload_to_wandb=False,
        record_env_id=0,
    ),
    headless_recording=True,
)

DEFAULTS = {
    "disabled": disabled,
    "wandb": wandb,
    "wandb_offline": wandb_offline,
    "debug_video": debug_video,
}
