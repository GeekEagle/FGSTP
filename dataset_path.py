class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'GasVid':
            return './datasets/GasVid'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
