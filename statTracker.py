import random

class StatTracker():

    """
    This class is a helper to track all the statistics of interest during training/val process
    """
    def __init__(self):
        
        self.data = {}
        return
    
    def add(self, epoch, episode, timeStep, stat, value):

        # 1 check if it is a new epoch
        if epoch not in self.data:
            self.data.setdefault(epoch, {})

        # 2 check if it is a new episode
        if episode not in self.data[epoch]:
            self.data[epoch].setdefault(episode, {})

        # 3 check if it is a new timeStep
        if timeStep not in self.data[epoch][episode]:
            self.data[epoch][episode].setdefault(timeStep, {})

        # 4 check if it is a new stat
        if stat not in self.data[epoch][episode][timeStep]:
            self.data[epoch][episode][timeStep].setdefault(stat, value)
        
        return
    
    def get(self, stat, epoch=None, episode=None, timeStep=None):
        
        if (epoch is None and episode is None and timeStep is None):
            return self._getL0(stat)

        if (episode is None and timeStep is None):
            return self._getL1(stat, epoch)

        if (timeStep is None):
            return self._getL2(stat, epoch,episode)

        return self._getL3(stat, epoch, episode, timeStep)
    
    def _getL3(self, stat, epoch, episode, timeStep):
        return self.data[epoch][episode][timeStep][stat]

    def _getL2(self, stat, epoch, episode):

        r = []
        for t in self.data[epoch][episode].keys():
            r.append(self._getL3(stat, epoch, episode, t))

        return r
    
    def _getL1(self, stat, epoch):

        r = []
        for epi in self.data[epoch].keys():
            r.append(self._getL2(stat,epoch,epi))

        return r

    def _getL0(self, stat):

        r = []
        for epoch in self.data.keys():
            r.append(self._getL1(stat, epoch))

        return r
    
class StatTracker2():

    """
    This class is a helper to track all the statistics of interest during training/val process
    """
    def __init__(self):
        
        self.data = {}
        return
    
    def add(self, epoch, episode, timeStep, stat, value):

        # 1 check if it is a new epoch
        if epoch not in self.data:
            self.data.setdefault(epoch, {})

        # 2 check if it is a new episode
        if episode not in self.data[epoch]:
            self.data[epoch].setdefault(episode, {})

        # 3 check if it is a new timeStep
        if timeStep not in self.data[epoch][episode]:
            self.data[epoch][episode].setdefault(timeStep, {})

        # 4 check if it is a new stat
        if stat not in self.data[epoch][episode][timeStep]:
            self.data[epoch][episode][timeStep].setdefault(stat, value)
        
        return
    
    def get(self, stat, epoch=None, episode=None, timeStep=None):
        
        if (epoch is None and episode is None and timeStep is None):
            return self._getL0(stat)

        if (episode is None and timeStep is None):
            return self._getL1(stat, epoch)

        if (timeStep is None):
            return self._getL2(stat, epoch,episode)

        return self._getL3(stat, epoch, episode, timeStep)
    
    def _getL3(self, stat, epoch, episode, timeStep):
        return self.data[epoch][episode][timeStep][stat]

    def _getL2(self, stat, epoch, episode):

        r = []
        for t in self.data[epoch][episode].keys():
            r.append(self._getL3(stat, epoch, episode, t))

        return r
    
    def _getL1(self, stat, epoch):

        r = []
        for epi in self.data[epoch].keys():
            r.append(self._getL2(stat,epoch,epi))

        return r

    def _getL0(self, stat):

        r = []
        for epoch in self.data.keys():
            r.append(self._getL1(stat, epoch))

        return r
if __name__ == '__main__':

    st = StatTracker()
    
    # set fake number of epochs, episodes and time steps.
    epochs = 2
    episodes = 3
    timeSteps = 4

    # fake some stats to fill the stat tracker object
    for e in range(epochs):
        for epi in range(episodes):
            for t in range(timeSteps):

                R = random.randint(0,10)
                l = random.random()

                st.add(e,epi,t,'reward', R)
                st.add(e,epi,t,'loss', l)


    # recover the reward for a particular epoch, episode and time step
    epoch = 1
    episode = 2
    t = 3
    r1 = st.get('reward', epoch, episode, t)
    print(r1)

    # recover the rewards for a particular epoch, episode
    # returns a number for each time step
    r2 = st.get('reward', epoch, episode)
    print(r2)

    # recover the rewards for a particular epoch
    # returns as many numbers as time steps have been done during all episodes
    r3 = st.get('reward', epoch)
    print(r3)

    # recover all rewards
    # a number per time step during all epochs and all episodes
    r4 = st.get('reward')
    print(r4)




    

