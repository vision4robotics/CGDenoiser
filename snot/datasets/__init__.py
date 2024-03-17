from .uavdark import UAVDARKDataset
from .darktrack import DARKTRACKDataset
from .cgd_rw import CGD_rw


datapath = {
            'UAVDark135':'/UAVDark135',
            'DarkTrack2021':'/DarkTrack2021',
            'CGD_rw':'/CGD_rw',
            }

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):

        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAVDark' in name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'DarkTrack2021' in name:
            dataset = DARKTRACKDataset(**kwargs)
        elif 'CGD_rw' in name:
            dataset = CGD_rw(**kwargs)
        
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset
