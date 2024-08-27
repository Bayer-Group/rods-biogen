import datetime
import warnings

from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "datenow", lambda: datetime.datetime.now().strftime("%Y%m%d%H%M")
)
OmegaConf.register_new_resolver(
    "id_suffix", lambda val: "" if val is None else f"_init{val}"
)


warnings.filterwarnings(action="ignore")
