import yaml

def get_args(config_path):
    """
    Read the YAML file and return the settings as a dictionary.

    Args:
        config_path (str): Path to the YAML file.
    
    Returns:
        dict: Dictionary of settings
    """
    with open(config_path, "r") as yaml_file:
        settings = yaml.safe_load(yaml_file)

    return settings

def update_yaml(file_path, **updates):
    """
    Update the YAML file with the provided settings.

    Args:
        file_path (str): Path to the YAML file.
        **updates: Dictionary of settings to update.

    Returns:
        None
    """
    my_args = get_args(file_path)

    # Apply updates
    my_args.update(updates)

    # Save back to file
    with open(file_path, 'w') as file:
        yaml.safe_dump(my_args, file)
