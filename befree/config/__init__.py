import yaml

def get_config(config_path):

    assert config_path.split('.')[-1] == 'yaml'
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
