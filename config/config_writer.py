import yaml

'''
Saves method names, tuning details of each method, 
training details of each method and testing details 
of each method

WIP
'''
def dump_data(data, filename='config'):
    with open(f'{filename}.yaml', 'w',) as f :
        yaml.dump(data, f, sort_keys=False) 
    print('Written to file successfully')

methods = ['standard_dropout', 'standard_dropconnect', 'mcdropout', 'mcdropconnect', 'flipout', 'duq', 'ensembles']



