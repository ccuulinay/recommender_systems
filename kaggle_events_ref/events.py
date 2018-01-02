from kaggle_events_ref.data_cleaner import DataCleaner
import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd
from sklearn.preprocessing import normalize


class Events:
    """
      构建event-event相似度，注意这里有2种相似度：
      1）由用户-event行为，类似协同过滤算出的相似度
      2）由event本身的内容(event信息)计算出的event-event相似度
      """

    def __init__(self, programEntities, psim=ssd.correlation, csim=ssd.cosine):
        cleaner = DataCleaner()
        fin = open("events.csv", 'rb')
        fin.readline()  # skip header
        nevents = len(programEntities.eventIndex.keys())
        self.eventPropMatrix = ss.dok_matrix((nevents, 7))
        self.eventContMatrix = ss.dok_matrix((nevents, 100))
        ln = 0
        for line in fin.readlines():
            #      if ln > 10:
            #        break
            cols = line.strip().split(",")
            eventId = cols[0]
            if programEntities.eventIndex.has_key(eventId):
                i = programEntities.eventIndex[eventId]
                self.eventPropMatrix[i, 0] = cleaner.getJoinedYearMonth(cols[2])  # start_time
                self.eventPropMatrix[i, 1] = cleaner.getFeatureHash(cols[3])  # city
                self.eventPropMatrix[i, 2] = cleaner.getFeatureHash(cols[4])  # state
                self.eventPropMatrix[i, 3] = cleaner.getFeatureHash(cols[5])  # zip
                self.eventPropMatrix[i, 4] = cleaner.getFeatureHash(cols[6])  # country
                self.eventPropMatrix[i, 5] = cleaner.getFloatValue(cols[7])  # lat
                self.eventPropMatrix[i, 6] = cleaner.getFloatValue(cols[8])  # lon
                for j in range(9, 109):
                    self.eventContMatrix[i, j - 9] = cols[j]
                ln += 1
        fin.close()
        self.eventPropMatrix = normalize(self.eventPropMatrix,
                                         norm="l1", axis=0, copy=False)
        sio.mmwrite("EV_eventPropMatrix", self.eventPropMatrix)
        self.eventContMatrix = normalize(self.eventContMatrix,
                                         norm="l1", axis=0, copy=False)
        sio.mmwrite("EV_eventContMatrix", self.eventContMatrix)
        # calculate similarity between event pairs based on the two matrices
        self.eventPropSim = ss.dok_matrix((nevents, nevents))
        self.eventContSim = ss.dok_matrix((nevents, nevents))
        for e1, e2 in programEntities.uniqueEventPairs:
            i = programEntities.eventIndex[e1]
            j = programEntities.eventIndex[e2]
            if not self.eventPropSim.has_key((i, j)):
                epsim = psim(self.eventPropMatrix.getrow(i).todense(),
                             self.eventPropMatrix.getrow(j).todense())
                self.eventPropSim[i, j] = epsim
                self.eventPropSim[j, i] = epsim
            if not self.eventContSim.has_key((i, j)):
                ecsim = csim(self.eventContMatrix.getrow(i).todense(),
                             self.eventContMatrix.getrow(j).todense())
                self.eventContSim[i, j] = epsim
                self.eventContSim[j, i] = epsim
        sio.mmwrite("EV_eventPropSim", self.eventPropSim)
        sio.mmwrite("EV_eventContSim", self.eventContSim)
