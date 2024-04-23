import numpy as np
import torch
import random
import collections
import math
import warnings
warnings.filterwarnings('ignore')

NEG_INF = -float("inf")

class CTC:
    def __init__(self, blank=0):

        """
        Args:
            blank: (int):
                    blank label index
        """
        self.blank = blank

    def standardize_label(self, label):
        """
        Args:
            label: (list):
                    (L, ), list of labels
        """
        L = len(label)
        new_label = [self.blank]
        for i in range(L):
            new_label.append(label[i])
            new_label.append(self.blank)
        return new_label
    
    def ctc_greedy_decode(self, logits, ignore = [1, 41, 42]):
        """
        Args:
            logits: (np.ndarray):
                    (T, V), output of the neural network, before softmax
            ignore: (list):
                    list of labels to ignore

        
        """
        argmaxes = np.argmax(logits, axis = -1)
        prev = argmaxes[0]
        out = []
        for i in range(1, argmaxes.shape[0]):
            if argmaxes[i] != prev:
                out.append(argmaxes[i])
            prev = argmaxes[i]
        out = [[x for x in out if x != self.blank and x not in ignore]]

        return out  
    def likelihood_ratio(self, y, total_prob):
        """
        Args:
            y:  (np.ndarray): 
                    (T, V), softmax of output of the neural network
            total_prob: (float):
                    total probability of the sequence, output of CTCforward
        """
        prob_most_likely_seq = y.max(axis=1).prod()
        print(total_prob, prob_most_likely_seq)
        return total_prob / prob_most_likely_seq
    def no_blank_normalization(self, y, total_prob):
        """
        Args:
            y:  (np.ndarray): 
                    (T, V), softmax of output of the neural network
            total_prob: (float):
                    total probability of the sequence, output of CTCforward
        """
        return np.exp(np.log(total_prob) / ((1 - y[:, self.blank]).sum()))

        #return np.exp(np.log(total_prob) / (y.shape[0] - np.where(np.argmax(y, axis=1) == self.blank)[0].shape[0]))

    def seg_length_normalization(self, y, total_prob):
        """
        Args:
            y:  (np.ndarray): 
                    (T, V), softmax of output of the neural network
            total_prob: (float):
                    total probability of the sequence, output of CTCforward
        """
        return np.exp(np.log(total_prob) / y.shape[0])
    
    def seg_length_normalization_and_ratio(self, y, total_prob):
        """
        Args:
            y:  (np.ndarray): 
                    (T, V), softmax of output of the neural network
            total_prob: (float):
                    total probability of the sequence, output of CTCforward
        """
        Cnf = self.seg_length_normalization(y, total_prob)
        Cnf_star = np.exp(np.log(y.max(axis=1).prod()) / y.shape[0])
        Cnf_r = Cnf / Cnf_star
        return Cnf_r
    
    def no_blank_normalization_and_ratio(self, y, total_prob):
        """
        Args:
            y:  (np.ndarray): 
                    (T, V), softmax of output of the neural network
            total_prob: (float):
                    total probability of the sequence, output of CTCforward
        """
        Cnf = self.no_blank_normalization(y, total_prob)
        Cnf_star = np.exp(np.log(y.max(axis=1).prod()) / ((1 - y[:, self.blank]).sum()))
        #Cnf_star = np.exp(np.log(y.max(axis=1).prod()) / (y.shape[0] - np.where(np.argmax(y, axis=1) == self.blank)[0].shape[0]))   
        Cnf_r = Cnf / Cnf_star
        return Cnf_r
    
    @staticmethod
    def make_new_beam():
        fn = lambda : (-float("inf"), -float("inf"))
        return collections.defaultdict(fn)
    @staticmethod
    def logsumexp(*args):
        """
        Stable log sum exp.
        """
        if all(a == NEG_INF for a in args):
            return NEG_INF
        a_max = max(args)
        lsp = math.log(sum(math.exp(a - a_max)
                        for a in args))
        return a_max + lsp
    
    def ctc_prefix_beam_search_decode(self, y, beam_size=10, n_best=1, blank=0):
        """
        CTC prefix beam search to get the most probable sequence of characters
        adapted from Awni Hannun's implementation: 
        https://github.com/HawkAaron/E2E-ASR/blob/master/ctc_decoder.py

        Args:
            y (np.ndarray): 
                (T, V): Softmax of output of the neural network, 
                where T is the length of the sequence and V is the number of classes
            beam_size (int): 
                Beam size for the beam search
            blank (int): 
                Index of the blank token
        Returns:
            out (List of int):
                List of decoded characters
        """
        T, V = y.shape
        log_y = np.log(y)
        beam = [(tuple(), (0.0, NEG_INF))]
        for t in range(T):
            next_beam = self.make_new_beam()
            for v in range(V):
                p = log_y[t, v]
                for prefix, (p_b, p_nb) in beam:
                    if v == blank:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_b = self.logsumexp(n_p_b, p_b+p, p_nb+p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
                        continue
                    end_t = prefix[-1] if prefix else None
                    n_prefix = prefix + (v,)
                    n_p_b, n_p_nb = next_beam[n_prefix]
                    if v != end_t:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p, p_nb + p)
                        next_beam[n_prefix] = (n_p_b, n_p_nb)
                    else:
                        n_p_nb = self.logsumexp(n_p_nb, p_b + p)
                        next_beam[n_prefix] = (n_p_b, n_p_nb)
                    if v == end_t:
                        n_p_b, n_p_nb = next_beam[prefix]
                        n_p_nb = self.logsumexp(n_p_nb, p_nb + p)
                        next_beam[prefix] = (n_p_b, n_p_nb)
            beam = sorted(next_beam.items(),key=lambda x : self.logsumexp(x[1][0],x[1][1]),reverse=True)
            beam = beam[:beam_size]
        best_n = beam[:n_best]
        preds = [list(x[0]) for x in best_n]
        scores = [self.logsumexp(x[1][0], x[1][1]) for x in best_n] # scores are in log space
        out = [(pred, score) for pred, score in zip(preds, scores)]

        return out
                            

    def CTCforward(self, y, label):
        """
        Args:
            y:  (np.ndarray):
                    (T, V), softmax of output of the neural network
            label: (list):
                    (L, ), standardized label sequence        
        """

        T, V = y.shape
        L = len(label)
        alpha = np.zeros((T, L))
        #y = np.log(y)
        #alpha = np.full((T, L), -np.inf)  # Initialize with -inf for log space

        alpha[0, 0] = y[0, label[0]]
        alpha[0, 1] = y[0, label[1]]
        for t in range(1, T):
            for l in range(L):
                if l >  2 * (t + 1) - 1:
                    continue
                else:
                    S = label[l]
                    #y_log = np.log(y[t, S])
                    if l == 0:
                        alpha[t, l] = alpha[t-1, l] 
                    elif l == 1:
                        #alpha[t, l] = np.logaddexp(alpha[t-1, l], alpha[t-1, l-1])
                        alpha[t, l] = alpha[t-1, l] + alpha[t-1, l-1]
                    else:
                        if S !=label[l-2]:
                            #alpha[t, l] = np.logaddexp(np.logaddexp(alpha[t-1, l], alpha[t-1, l-1]), alpha[t-1, l-2])
                            alpha[t, l] = alpha[t-1, l] + alpha[t-1, l-1] + alpha[t-1, l-2]
                        else:
                            #alpha[t, l] = np.logaddexp(alpha[t-1, l], alpha[t-1, l-1])
                            alpha[t, l] = alpha[t-1, l] + alpha[t-1, l-1]
                    alpha[t, l] *= y[t, S] 
                    #alpha[t, l] += y[t, S]

        #alpha = np.exp(alpha)
        #return np.exp(alpha)

        total_prob = alpha[-1, -1] + alpha[-1, -2]
        return alpha, total_prob
    



"""
====================================================================================
Ignore the code below. It is used for testing the correctness of the implementation.
====================================================================================

"""

def reference_CTCforward(y, labels):
    T, V = y.shape
    L = len(labels)
    alpha = np.zeros([T, L])

    # init
    alpha[0, 0] = y[0, labels[0]]
    alpha[0, 1] = y[0, labels[1]]

    for t in range(1, T):
        for i in range(L):
            s = labels[i]
            
            a = alpha[t - 1, i] 
            if i - 1 >= 0:
                a += alpha[t - 1, i - 1]
            if i - 2 >= 0 and s != 0 and s != labels[i - 2]:
                a += alpha[t - 1, i - 2]
                
            alpha[t, i] = a * y[t, s]
            
    return alpha


if __name__ == "__main__":
    #torch.manual_seed(1)
    dict = {'-': 0, 'a': 1, 'b': 2, 'c': 3}
    ctc = CTC()

    for i in range(10):
        y = torch.rand(20, 4)
        y_log_softmax = torch.nn.functional.log_softmax(y, dim=1)
        y_log_softmax = y_log_softmax.numpy()
        y = torch.nn.functional.softmax(y, dim=1)
        y = y.numpy()
        label = np.random.randint(1, 4, (random.randint(5, 17 ), )).tolist()
        label = ctc.standardize_label(label)
        alpha = ctc.CTCforward(y_log_softmax, label)
        ref_alpha = reference_CTCforward(y, label)
        error = np.abs(alpha - ref_alpha).sum()
        print(f"Error: {error:.4f}")
        if error > 1e-3:
            print("label:", label)
        
            print("My implementation:")
            print(alpha.round(2).T)
            print("Reference implementation:")
            print(ref_alpha.round(2).T)
            print("Difference:")
            print(alpha.round(4).T - ref_alpha.round(4).T)
            print("=" * 50)
            break

 