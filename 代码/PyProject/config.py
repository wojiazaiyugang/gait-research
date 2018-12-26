# 目录
PATH = r"D:\OneDrive - business\beihang university\研一上\移动计算\PyProject\temp"
# PATH = r"C:\Users\wojiazaiyugang\OneDrive - business\beihang university\研一上\移动计算\PyProject\temp"
# 文件
TEMP_FILE = r"/temp.txt"
# 每次采样间隔，单位毫秒
SAMPLE_DELAY = 20
# 一个间隔的时长，单位毫秒
DURATION_LENGTH = 3000
# 计算开始的组数（位置）
START_POS = 0
# 使用的组数
DATA_NUMBER = 100
# 步态识别模板起始位置
REFERENCE_BEGIN = 50
# 步态识别模板长度 100 --> 2S钟
REFERENCE_LENGTH = 100
# 使用的统计指标
STATISTICS = [
    # 注意，此数据中的特征顺序与all_people_result中的结果一一对应 ，因此此数组不能修改
    "x_average", "y_average", "z_average",
    "x_median", "y_median", "z_median",
    "average_resultant_acceleration",
    "x_standard_deviation", "y_standard_deviation", "z_standard_deviation",
    "x_skewness", "y_skewness", "z_skewness",
    "x_kurtosis", "y_kurtosis", "z_kurtosis",
    "x_up_quartile_deviation", "y_up_quartile_deviation", "z_up_quartile_deviation",
    "x_low_quartile_deviation", "y_low_quartile_deviation", "z_low_quartile_deviation",
    "step_duration",
    "step_max_distance",
    "step_min_distance",
]
# 用于尝试所有可能性，寻找的特征数组
TRY_STATISTICS = ["z_up_quartile_deviation", "x_skewness", "x_kurtosis", "y_kurtosis", "average_resultant_acceleration",
                  "x_up_quartile_deviation", "x_average", "y_low_quartile_deviation", "y_median",
                  "y_standard_deviation", "x_low_quartile_deviation", "x_standard_deviation", "x_median", "y_average",
                  "z_standard_deviation", "step_max_distance", "y_up_quartile_deviation", "step_duration", "z_median",
                  "step_min_distance", "z_average", "z_low_quartile_deviation", "z_kurtosis", "y_skewness",
                  "z_skewness"]
# 最终选取的有效特征
FINAL_STATISTIC = ["y_up_quartile_deviation","y_low_quartile_deviation","y_average","x_up_quartile_deviation","z_median","y_median","z_average"]

# 所有的数据文件夹
DATA_DIRS = ["/解晓政/1", "/李辉勇", "", "/沥青酱/5", "/林佳萍", "/齐之平", "/任璐",
             "/于剑楠", "/赵青娟", "/程浩", "/欧阳"]

# 绘图使用的颜色
COLORS = ["r", "g", "b", "k", "m", "y", "c", "#fdfdfd", "#cacacc"]

# 当两个步态向量的距离小于这个阈值的时候，就认为两个步态向量足够相似了
THRESHOLD = 2.5
# 连续的多少个测试通过就开始进行更新
CONTINUOUS_PASS_NUMBER = 5
# 连续的多少个向量就认为稳定从而获得模板,实验1是3，实验2是2
CONTINUOUS_VECTOR = 2
# 俩参数
C1 = 0.8
C2 =0.2