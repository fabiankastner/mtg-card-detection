import configparser

def get_config():
    config = configparser.ConfigParser()
    config.sections()
    config.read('config.ini')
    return config