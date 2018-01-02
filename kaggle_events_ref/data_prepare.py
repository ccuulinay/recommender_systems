from kaggle_events_ref.util import ProgramEntities
from kaggle_events_ref.users import Users
from kaggle_events_ref.users_friends import UserFriends
from kaggle_events_ref.events import Events
from kaggle_events_ref.events_attendees import EventAttendees


def data_prepare():
  """
  计算生成所有的数据，用矩阵或者其他形式存储方便后续提取特征和建模
  """
  print("第1步：统计user和event相关信息...")
  pe = ProgramEntities()
  print("第1步完成...\n")
  print("第2步：计算用户相似度信息，并用矩阵形式存储...")
  Users(pe)
  print("第(2步完成...\n")
  print("第3步：计算用户社交关系信息，并存储...")
  UserFriends(pe)
  print("第3步完成...\n")
  print("第4步：计算event相似度信息，并用矩阵形式存储...")
  Events(pe)
  print("第4步完成...\n")
  print("第5步：计算event热度信息...")
  EventAttendees(pe)
  print("第5步完成...\n")

# 运行进行数据准备
data_prepare()
