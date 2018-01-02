import pandas as pd
import numpy as np


def test(clf):
  """
  读取test数据，用分类器完成预测
  """
  origTestDf = pd.read_csv("test.csv")
  users = origTestDf.user
  events = origTestDf.event
  testDf = pd.read_csv("data_test.csv")
  fout = open("result.csv", 'wb')
  fout.write(",".join(["user", "event", "outcome", "dist"]) + "\n")
  nrows = len(testDf)
  Xp = np.matrix(testDf)
  yp = np.zeros((nrows, 2))
  for i in range(0, nrows):
    xp = Xp[i, :]
    yp[i, 0] = clf.predict(xp)
    yp[i, 1] = clf.decision_function(xp)
    fout.write(",".join(map(lambda x: str(x),
      [users[i], events[i], yp[i, 0], yp[i, 1]])) + "\n")
  fout.close()
