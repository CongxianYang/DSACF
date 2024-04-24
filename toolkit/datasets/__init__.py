from .otb import OTBDataset
from .uav import UAVDataset
from .lasot import LaSOTDataset
from .got10k import GOT10kDataset
from toolkit.RGBT_datasets import gtot
from toolkit.RGBT_datasets import rgbt234
from toolkit.RGBT_datasets import rgbt234_lasher



class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'OTB2015', 'LaSOT', 'UAV123', 'NFS240', 'NFS30',
                'VOT2018', 'VOT2016', 'VOT2018-LT'
            dataset_root: dataset root
            load_img: wether to load image
          ##  list.file
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        root_dir=kwargs['dataset_root']
        if 'OTB' in name:
            dataset = OTBDataset(**kwargs)
        elif 'LaSOT' == name:
            dataset = LaSOTDataset(**kwargs)
        elif 'UAV' in name:
            dataset = UAVDataset(**kwargs)
        elif 'GOT-10k' == name:
            dataset = GOT10kDataset(**kwargs)
        elif 'RGBT234' == name:
            dataset = rgbt234.RGBT234(root_dir)
        elif 'GTOT' in name:
            dataset = gtot.GTOT(root_dir)
        elif 'LasHeR' == name:
            dataset = rgbt234_lasher.RGBT234_Lasher(root_dir)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

