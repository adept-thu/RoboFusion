class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Corruptions_mode(metaclass=Singleton):
    def __init__(self) -> None:
        print('Corruptions_mode init')
        self.corruption_type_l = None
        self.corruption_type_c = None
        self.severity = 0
        self.isOffline = False
        self.save_flag = False
            
    def set_corruption(self, corruption_type_l=None, corruption_type_c=None,severity=0) -> None:
        if severity==0:
            self.corruption_type_l = None
            self.corruption_type_c = None
        else:
            self.corruption_type_l = corruption_type_l
            self.corruption_type_c = corruption_type_c
        self.severity = severity
        
    def get_corruption(self):
        return self.corruption_type_l, self.corruption_type_c, self.severity
    
    def set_offline_flag(self, isOffline):
        self.isOffline = isOffline
    def get_offline_flag(self):
        return self.isOffline
        
    def set_save_flag(self, save_flag):
        if self.isOffline and save_flag:
            # raise ValueError('---'*10 + '离线数据不保存！' + '---'*10)
            print('---'*10 + '读取离线噪声数据集，不需要再保存该噪声数据集！' + '---'*10)
            self.save_flag = False
        self.save_flag = save_flag
        
    def get_save_flag(self):
        return self.save_flag
    
    def set_model_n(self, model_n):
        self.model_n = model_n
        
    def get_model_n(self):
        try:
            return self.model_n
        except:
            return None

    