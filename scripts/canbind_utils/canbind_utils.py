import yaml

def scale_name_mapping(yaml_file, col_names, scale_metadata):
    """
    TODO
    """
    # read yaml file with variable name mappings
    with open(yaml_file, 'r') as f:
        scale_name_json = yaml.load(f, Loader=yaml.FullLoader)
    item_dict = scale_name_json['items']

    # some variables have aliases that we have to consider
    items_w_alias = scale_metadata.loc[scale_metadata['Aliases'].notna()]

    # dictionary with primary variable as key, comma-sep aliases as values
    alias_dict = {item['ElementName']:item['Aliases'] for i,item in items_w_alias.iterrows()}

    # map the descriptive variables to each of the alises 
    item_dict = add_aliases_to_item_dict(item_dict, alias_dict)

    # map scale items to their descriptive names    
    rename_dict  = match_items_to_newnames(col_names, item_dict)
    return rename_dict


