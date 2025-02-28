from pathlib import Path


class DataPath:
    def __init__(self):
        # 获取当前文件的绝对路径
        self.current_file_path = Path(__file__).resolve()
        # 获取当前文件所在的目录路径
        self.current_directory = self.current_file_path.parent.parent.parent.parent
        self.data_directory = self.current_directory / Path("data") / Path("SJTU") / Path("DDQN")
        self.data_sansuo_directory = self.data_directory / Path("data_sansuo")
        self.data_sjtu_directory = self.data_directory / Path("data_sjtu")
        # 数据路径修改
        # self.data_sjtu_directory = self.data_directory / Path("data_zjnu")
        self.data_zjnu_directory = self.current_directory / Path("data") / Path("ZJNU")


dataPath = DataPath()
