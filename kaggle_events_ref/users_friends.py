import numpy as np
import scipy.io as sio
import scipy.sparse as ss
from sklearn.preprocessing import normalize

class UserFriends:
    """
    找出某用户的那些朋友，想法非常简单
    1)如果你有更多的朋友，可能你性格外向，更容易参加各种活动
    2)如果你朋友会参加某个活动，可能你也会跟随去参加一下
    """

    def __init__(self, programEntities):
        nusers = len(programEntities.userIndex.keys())
        self.numFriends = np.zeros((nusers))
        self.userFriends = ss.dok_matrix((nusers, nusers))
        fin = open("user_friends.csv", 'rb')
        fin.readline()  # skip header
        ln = 0
        for line in fin:
            if ln % 200 == 0:
                print("Loading line: ", ln)
            cols = line.strip().split(",")
            user = cols[0]
            if programEntities.userIndex.has_key(user):
                friends = cols[1].split(" ")
                i = programEntities.userIndex[user]
                self.numFriends[i] = len(friends)
                for friend in friends:
                    if programEntities.userIndex.has_key(friend):
                        j = programEntities.userIndex[friend]
                        # the objective of this score is to infer the degree to
                        # and direction in which this friend will influence the
                        # user's decision, so we sum the user/event score for
                        # this user across all training events.
                        eventsForUser = programEntities.userEventScores.getrow(j).todense()
                        score = eventsForUser.sum() / np.shape(eventsForUser)[1]
                        self.userFriends[i, j] += score
                        self.userFriends[j, i] += score
            ln += 1
        fin.close()
        # 归一化数组
        sumNumFriends = self.numFriends.sum(axis=0)
        self.numFriends = self.numFriends / sumNumFriends
        sio.mmwrite("UF_numFriends", np.matrix(self.numFriends))
        self.userFriends = normalize(self.userFriends, norm="l1", axis=0, copy=False)
        sio.mmwrite("UF_userFriends", self.userFriends)
