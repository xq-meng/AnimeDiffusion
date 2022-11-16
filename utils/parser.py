def update_config(config: dict, enhance_parse: dict):
    if enhance_parse is None:
        return
    for k, v in enhance_parse.items():
        if type(v) == dict:
            update_config(config=config[k], enhance_parse=enhance_parse[k])
        else:
            config.update(enhance_parse)