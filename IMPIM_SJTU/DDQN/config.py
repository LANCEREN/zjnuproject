from pathlib import Path  


class DataPath():
    def __init__(self):
        # 获取当前文件的绝对路径
        self.current_file_path = Path(__file__).resolve()  
        # 获取当前文件所在的目录路径
        self.current_directory = self.current_file_path.parent
        self.data_directory = self.current_directory / Path('data')
        self.data_sansuo_directory = self.data_directory / Path('data_sansuo')
        self.data_sjtu_directory = self.data_directory / Path('data_sjtu')
        self.data_zjnu_directory = self.data_directory / Path('data_zjnu')

dataPath = DataPath()