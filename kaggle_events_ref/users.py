
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd
from sklearn.preprocessing import normalize
from kaggle_events_ref.data_cleaner import DataCleaner


class Users(object):
    """
    构建 user/user 相似度矩阵
    """

    def __init__(self, programEntities, sim=ssd.correlation):
        cleaner = DataCleaner()
        nusers = len(programEntities.userIndex.keys())
        fin = open("../data/users.csv", 'rb')
        colnames = fin.readline().strip().split(",")
        self.userMatrix = ss.dok_matrix((nusers, len(colnames) - 1))
        for line in fin:
            cols = line.strip().split(",")
            # 只考虑train.csv中出现的用户
            if programEntities.userIndex.has_key(cols[0]):
                i = programEntities.userIndex[cols[0]]
                ##将数据进行预处理再放进userMatrix矩阵中
                self.userMatrix[i, 0] = cleaner.getLocaleId(cols[1])
                self.userMatrix[i, 1] = cleaner.getBirthYearInt(cols[2])
                self.userMatrix[i, 2] = cleaner.getGenderId(cols[3])
                self.userMatrix[i, 3] = cleaner.getJoinedYearMonth(cols[4])
                self.userMatrix[i, 4] = cleaner.getCountryId(cols[5])
                self.userMatrix[i, 5] = cleaner.getTimezoneInt(cols[6])
        fin.close()
        # 归一化用户矩阵
        self.userMatrix = normalize(self.userMatrix, norm="l1", axis=0, copy=False)
        sio.mmwrite("US_userMatrix", self.userMatrix)
        # 计算用户相似度矩阵，之后会用到
        self.userSimMatrix = ss.dok_matrix((nusers, nusers))
        for i in range(0, nusers):
            self.userSimMatrix[i, i] = 1.0
        for u1, u2 in programEntities.uniqueUserPairs:
            i = programEntities.userIndex[u1]
            j = programEntities.userIndex[u2]
            if not self.userSimMatrix.has_key((i, j)):
                usim = sim(self.userMatrix.getrow(i).todense(),
                           self.userMatrix.getrow(j).todense())
                self.userSimMatrix[i, j] = usim
                self.userSimMatrix[j, i] = usim
        sio.mmwrite("US_userSimMatrix", self.userSimMatrix)
