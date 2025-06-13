from models.pipvit import get_network
from models.pipvit import PiPViT

def create_model(model_config, logger):
    model_type = model_config['model_type'].replace('_normed_relu', '')

    logger.info('[INFO] 2D maps are normed2D.'
                'Backbone is set to {}'.format(model_config['model_type']))
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes,feature_norm_layer = get_network(num_classes=model_config['num_classes'], model_name=model_type,
                        imgnet_pretrained=model_config['imgnet_pretrained'], num_features=0, bias=False,
                   img_size=model_config['image_size'])

    model = PiPViT(num_classes=model_config['num_classes'],
                 num_prototypes=num_prototypes,
                 feature_net=feature_net,
                   num_features=0,
                 add_on_layers=add_on_layers,
                 pool_layer=pool_layer,
                   _norm_layer=feature_norm_layer,
                 classification_layer=classification_layer)
    return model



