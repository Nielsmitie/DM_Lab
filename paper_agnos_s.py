import json
import os

from pipeline import main


def run_hardcode_id():
    with open(os.path.join('configs', 'agnos_s_config.json')) as fr:
        config = json.load(fr)

    for dataset, id in zip(['TOX-171', 'warpPIE10P', 'Yale'],
                 [40, 9, 6, 4, 23, 15, 3, 10]):
        config['dataset'][config['pipeline']['dataset']]['name'] = dataset
        config['id']['hardcode']['id'] = id

        print('running agnos s with {} and id {}'.format(dataset, id))
        main([], config)


if __name__ == '__main__':
    run_hardcode_id()
