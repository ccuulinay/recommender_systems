from collections import defaultdict
import itertools
import pickle
import scipy.io as sio
import scipy.sparse as ss


class ProgramEntities(object):
    """
  我们只关心train和test中出现的user和event，因此重点处理这部分关联数据
  """

    def __init__(self):
        # 统计训练集中有多少独立的用户的events
        uniqueUsers = set()  ##统计users
        uniqueEvents = set()  ##统计events
        eventsForUser = defaultdict(set)  ##统计{event:set(users)}
        usersForEvent = defaultdict(set)  ##统计{user:set(events)}
        for filename in ["../data/train.csv", "../data/test.csv"]:
            f = open(filename, 'rb')
            f.readline().strip().split(",")
            for line in f:
                cols = line.strip().split(",")
                uniqueUsers.add(cols[0])
                uniqueEvents.add(cols[1])
                eventsForUser[cols[0]].add(cols[1])
                usersForEvent[cols[1]].add(cols[0])
            f.close()
        self.userEventScores = ss.dok_matrix((len(uniqueUsers), len(uniqueEvents)))  ##统计各个user对各个event感兴趣程度
        self.userIndex = dict()
        self.eventIndex = dict()
        for i, u in enumerate(uniqueUsers):
            self.userIndex[u] = i
        for i, e in enumerate(uniqueEvents):
            self.eventIndex[e] = i
        ftrain = open("../data/train.csv", 'rb')
        ftrain.readline()
        for line in ftrain:
            cols = line.strip().split(",")
            i = self.userIndex[cols[0]]
            j = self.eventIndex[cols[1]]
            self.userEventScores[i, j] = int(cols[4]) - int \
                (cols[5])  ##统计train中各个user对各个event感兴趣程度;cols[4]为interest列;cols[5]为not interested列
        ftrain.close()
        sio.mmwrite("PE_userEventScores", self.userEventScores)
        # 为了防止不必要的计算，我们找出来所有关联的user 或者 关联的event
        # 所谓的关联用户，指的是至少在同一个event上有行为的用户pair
        # 关联的event指的是至少同一个user有行为的event pair
        self.uniqueUserPairs = set()
        self.uniqueEventPairs = set()
        for event in uniqueEvents:
            users = usersForEvent[event]
            if len(users) > 2:
                self.uniqueUserPairs.update(itertools.combinations(users,
                                                                   2))  ##itertools.combinations(users, 2)表示在users中随机抽取两个不同的user进行组合，然后更新到uniqueUserPairs。
        for user in uniqueUsers:
            events = eventsForUser[user]
            if len(events) > 2:
                self.uniqueEventPairs.update(itertools.combinations(events, 2))
        pickle.dump(self.userIndex, open("PE_userIndex.pkl", 'wb'))
        pickle.dump(self.eventIndex, open("PE_eventIndex.pkl", 'wb'))
