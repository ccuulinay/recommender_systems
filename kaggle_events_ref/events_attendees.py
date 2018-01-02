import scipy.io as sio
import scipy.sparse as ss
import scipy.spatial.distance as ssd
from sklearn.preprocessing import normalize

class EventAttendees():
    """
  统计某个活动，参加和不参加的人数，从而为活动活跃度做准备
  """

    def __init__(self, programEvents):
        nevents = len(programEvents.eventIndex.keys())
        self.eventPopularity = ss.dok_matrix((nevents, 1))
        f = open("event_attendees.csv", 'rb')
        f.readline()  # skip header
        for line in f:
            cols = line.strip().split(",")
            eventId = cols[0]
            if programEvents.eventIndex.has_key(eventId):
                i = programEvents.eventIndex[eventId]
                self.eventPopularity[i, 0] = \
                    len(cols[1].split(" ")) - len(cols[4].split(" "))
        f.close()
        self.eventPopularity = normalize(self.eventPopularity, norm="l1",
                                         axis=0, copy=False)
        sio.mmwrite("EA_eventPopularity", self.eventPopularity)
