#Import needed libraryies
import scipy
import numpy as np
import matplotlib.pyplot as plt
import math

#Implement class based on signal detection theory
class SignalDetection:
    #Class constructor
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        #signal detection theory variables
        self.hits = hits 
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections
        #Check for invalid input
        if(self.__variable_isInvalid()):
            raise ValueError("SDT values must be positive!")

    #class method to get the hit rate: hits/total signal trials
    def hit_rate(self):
        #Check for Corruption
        if(self.__variable_isInvalid()):
            raise ValueError("SDT values must be positive!")
        return self.hits / (self.hits + self.misses)

    #class method to get the false alarm rate: false alarms/total noise trials
    def falseAlarm_rate(self):
        #Check for Corruption
        if(self.__variable_isInvalid()):
            raise ValueError("SDT values must be positive!")
        return self.falseAlarms / (self.falseAlarms + self.correctRejections)

    #class method to calculate d': Z(H) - Z(FA), Z = stats.norm.ppf
    def d_prime(self):
        #get z values
        z_hit = scipy.stats.norm.ppf(self.hit_rate()) 
        z_falseAlarm = scipy.stats.norm.ppf(self.falseAlarm_rate())
        #return d'
        return z_hit-z_falseAlarm

    #class method to get the criterion: -0.5*(Z(H) + Z(FA)), Z = stats.norm.ppf
    def criterion(self):
        #get Z values
        z_hit = scipy.stats.norm.ppf(self.hit_rate())
        z_falseAlarm = scipy.stats.norm.ppf(self.falseAlarm_rate())
        #return criterion
        return -0.5*(z_hit + z_falseAlarm)
    
    #add two sdt class object together
    def __add__(self, other):
        if(self.__variable_isInvalid() or other.__variable_isInvalid()):
            raise ValueError("SDT values must be positive!")
        return SignalDetection(self.hits + other.hits, self.misses + other.misses, self.falseAlarms + other.falseAlarms, self.correctRejections + other.correctRejections)

    #multiply a sdt class object by a constant
    def __mul__(self, scalar):
        if(self.__variable_isInvalid()):
            raise ValueError("SDT values must be positive!")
        return SignalDetection(self.hits * scalar, self.misses * scalar, self.falseAlarms * scalar, self.correctRejections * scalar)

    #plot the reciever operating characteristic (roc) curve for the sdt class object
    def plot_roc(self):
        #get the range of the sdt class object to look at
        threshold_list = np.arange(-10, 10, 0.01)
        x_points = []
        y_points = []
        #get FP and TP at varying threholds to plot the curve
        for threshold in threshold_list:
            FP = 1-scipy.stats.norm.cdf(threshold, 0, 1) #noise curve
            TP = 1-scipy.stats.norm.cdf(threshold, self.d_prime(), 1) #signal curve
            x_points.append(FP)
            y_points.append(TP)
        #plot the curve
        plt.plot(x_points, y_points, '-', linewidth = 2)
        plt.plot([self.falseAlarm_rate()], [self.hit_rate()], 'o')
        plt.annotate("Classifying Threshold Point", xy = (self.falseAlarm_rate(), self.hit_rate()),
                    xytext=(self.falseAlarm_rate()+0.1, self.hit_rate()-0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05))
        plt.plot([0, 1], [0, 1], '--')
        #label the graph
        plt.title('Receiver Operating Characteristic')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Hit Rate')
        plt.xlabel('False Alarm Rate')
        plt.show()

    #Plot the signal detection theory plot for the given object
    def plot_sdt(self):
        x = np.arange(-5, 5, 0.01)
        x_d_prime_line = np.arange(0, self.d_prime(), 0.01)
        #noise curve
        plt.plot(x, scipy.stats.norm.pdf(x, 0, 1))
        plt.annotate("Noise Curve", xy = (x[400], scipy.stats.norm.pdf(x[400], 0, 1)), 
                     xytext = (x[400]-3, scipy.stats.norm.pdf(x[400], 0, 1)+0.1), 
                     arrowprops=dict(facecolor='black', shrink=0.05), 
                     fontsize=15)
        #signal curve
        plt.plot(x, scipy.stats.norm.pdf(x, self.d_prime(), 1))
        annotate_index = round(999*(0.55 + self.d_prime()/10))
        plt.annotate("Signal Curve", xy = (x[annotate_index], scipy.stats.norm.pdf(x[annotate_index], self.d_prime(), 1)), 
                     xytext = (x[annotate_index]+0.5, scipy.stats.norm.pdf(x[annotate_index], self.d_prime(), 1)+0.05), 
                     arrowprops=dict(facecolor='black', shrink=0.05), 
                     fontsize=15)
        #Threshold line
        top_of_c = max(scipy.stats.norm.pdf(self.d_prime()/2 + self.criterion(), 0, 1), scipy.stats.norm.pdf(self.d_prime()/2 + self.criterion(), self.d_prime(), 1))
        plt.plot([self.d_prime()/2 + self.criterion(), self.d_prime()/2 + self.criterion()], [0, top_of_c], '-', color = "red")
        plt.annotate("threshold", xy = (self.d_prime()/2 + self.criterion()+0.05, top_of_c/4), fontsize = 15)
        #d' line
        plt.plot([0, self.d_prime()], [0.4, 0.4], '-', color = "green")
        plt.annotate("d'", xy=(self.d_prime()/2, 0.41), fontsize = 15)
        #label the graph
        plt.xlabel("Decision Label")
        plt.ylabel("Probability")
        plt.title("Signal Detection Theory Plot")
        plt.ylim(0, 0.5)
        plt.show()

    #private method to test if all variables in the class object are positive
    def __variable_isInvalid(self):
        #check condition
        if(self.hits <= 0 or 
           self.misses <= 0 or
           self.falseAlarms <= 0 or
           self.correctRejections <= 0):
            return(True)
        #otherwise, all variables are valid
        return(False)

    #simulate and return a list of multiple sdt objects
    @staticmethod
    def simulate(dPrime, criteriaList, signalCount, noiseCount):
        if(signalCount <= 0 or noiseCount <= 0):
            raise ValueError("Error, only positive values allowed for signal/noise count!")
        s_list = []
        for i in range(len(criteriaList)):
            k = criteriaList[i] + (dPrime / 2)
            hit_rate = 1 - scipy.stats.norm.cdf(k - dPrime)
            false_alarm_rate = 1 - scipy.stats.norm.cdf(k)
            hits = np.random.binomial(signalCount, hit_rate)
            misses = signalCount - hits
            falseAlarm = np.random.binomial(noiseCount, false_alarm_rate)
            correctRejections = noiseCount - falseAlarm
            s_list.append(SignalDetection(hits, misses, falseAlarm, correctRejections))
        return s_list

    #plot the best fit roc from a list of sdt objects
    @staticmethod
    def plot_roc(sdtList):
        x = list()
        y = list()
        for i in range(len(sdtList)):
            if(sdtList[i].__variable_isInvalid()):
                raise ValueError("SDT values must be positive!")
            x.append(sdtList[i].falseAlarm_rate())
            y.append(sdtList[i].hit_rate())
        plt.scatter(x, y, c='k')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit rate")
        plt.title("ROC curve")

    #return the log likliehood of the datapoint based on a theoretical hit and false alarm rate
    def nLogLikelihood(self, hit_rate, falseAlarm_rate):
        if(hit_rate < 0 or falseAlarm_rate < 0):
            raise ValueError("Hit and flase alarm rate must be non-negative")
        likelihood = (-(self.hits) * (math.log(hit_rate))) - (self.misses * (math.log(1 - hit_rate))) - (self.falseAlarms * (math.log(falseAlarm_rate))) - (self.correctRejections * (math.log(1 - falseAlarm_rate)))
        return(likelihood)

    #define function for roc curve
    @staticmethod
    def rocCurve(falseAlarmRate, a):
        try:
            for rate in falseAlarmRate:
                if(rate < 0):
                    raise ValueError("False alarm rate must be a non-negative value")
        except:
            if(falseAlarmRate < 0):
                    raise ValueError("False alarm rate must be a non-negative value")
        #compute the predicted hit rate based on false alarm rate
        hitRate = scipy.stats.norm.cdf(a + scipy.stats.norm.ppf(falseAlarmRate))
        return(hitRate)
    
    #define roc loss function
    @staticmethod
    def rocLoss(a, sdtList):
        #calculate the log-liklihood for each predicted hit rate to the observed false alarm rate
        loss = 0
        for sdt in sdtList:
            loss += sdt.nLogLikelihood(SignalDetection.rocCurve(sdt.falseAlarm_rate(), a), sdt.falseAlarm_rate())
        return(loss)
    
    #fit the roc plot to a list of sdt objects
    @staticmethod
    def fit_roc(sdtList):
        #get the optimal value for a that minimuzes the loss function
        optimization_result = scipy.optimize.minimize(fun = SignalDetection.rocLoss, x0 = 0, args = (sdtList,))
        #result is a_hat which should be close to d'
        a_hat = optimization_result.x[0]
        #Get the fitted data to plot
        falsesAlarms = np.arange(0, 1, 0.01)
        hitRates = SignalDetection.rocCurve(falsesAlarms, a_hat)
        #plot the fitted cirve
        plt.plot(falsesAlarms, hitRates, 'r')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit rate")
        plt.title("ROC curve")
        plt.show()
        return(a_hat)