from jaxtyping import Float, Array, PRNGKeyArray

from jepax.model.ijepa import IJEPA, IJEPAEncoder, IJEPAPredictor,\
      get_ijepa_config, get_ijepa_model, get_encoder_config, get_predictor_config
from jepax.model.vit import ViTclassifier, vit_classifier_configs,\
      get_vit_config, get_vit_clf_model