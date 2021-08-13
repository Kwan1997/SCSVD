from createRatingMatrix import create_ratings
from createTestingMatrix import create_testings
from createModularityMatrix import create_mMatrix
from math import inf
from math import sqrt
from math import fabs
from predictions import Prediction
from eval_utils import *


class SCMF(object):
    def __init__(self, rpath, vpath, tpath, rmin, rmax):
        self.tpath = tpath
        self.rating_scale = (float(rmin), float(rmax))
        self.initMean = 0.0
        self.initStd = 0.1
        self.numFactors = 10  # f
        self.numCommunities = 300  # c
        self.alpha = 25  # Map from U to H
        self.beta = 100  # Modularity
        self.eta = 125  # affinity between F and H
        self.gamma = 0.03
        self.learnrate = 0.01  # learning rate
        self.maxiteration = 40
        self.impitem = 0.03
        self.threshold = 3.0
        self.trainset = create_ratings(rpath, self.rating_scale)
        self.testset = create_testings(vpath, self.rating_scale)
        self.writefile = False  # write txt or not

    def print_parameters(self):
        print('*' * 40)
        print('rating scale is: ' + str(self.rating_scale))
        print('number of factors: ' + str(self.numFactors))
        print('number of communities: ' + str(self.numCommunities))
        print('alpha: ' + str(self.alpha))
        print('beta: ' + str(self.beta))
        print('eta: ' + str(self.eta))
        print('gamma: ' + str(self.gamma))
        print('implicit item factor: ' + str(self.impitem))
        print('learning rate: ' + str(self.learnrate))
        print('maximum number of iteration: ' + str(self.maxiteration))
        print('*' * 40)

    def setup(self):
        B, A = create_mMatrix(self.trainset, self.tpath)
        B = np.multiply(B, A)
        U = np.random.normal(self.initMean, self.initStd, (self.numFactors, self.trainset.n_users))
        V = np.random.normal(self.initMean, self.initStd, (self.numFactors, self.trainset.n_items))
        user_b = np.random.normal(self.initMean, self.initStd, self.trainset.n_users)
        item_b = np.random.normal(self.initMean, self.initStd, self.trainset.n_items)
        Imp = np.random.normal(self.initMean, self.initStd, (self.numFactors, self.trainset.n_items))
        H = np.random.normal(1 / self.numCommunities, self.initStd, (self.trainset.n_users, self.numCommunities))
        C = np.random.normal(self.initMean, self.initStd, (self.numFactors, self.numCommunities))
        F = np.random.normal(1 / self.numCommunities, self.initStd, (self.trainset.n_users, self.numCommunities))
        # F = H
        Inc = np.eye(self.trainset.n_users, self.numCommunities)
        return B, A, U, V, user_b, item_b, Imp, H, C, F, Inc

    def train(self):
        np.random.seed(5)
        self.print_parameters()
        B, A, U, V, user_b, item_b, Imp, H, C, F, Inc = self.setup()
        last_loss = inf
        for step in range(self.maxiteration):
            if (step + 1) % 10 == 0:
                self.learnrate *= 0.8
            # print(step+1, self.learnrate)
            CCT = C @ C.transpose()
            CHT = C @ H.transpose()
            objective = 0
            objective += (self.gamma / 2) * (
                    np.linalg.norm(U, 'fro') ** 2 + np.linalg.norm(V, 'fro') ** 2 + np.linalg.norm(Imp,
                                                                                                   'fro') ** 2 + np.linalg.norm(
                user_b, 2) ** 2 + np.linalg.norm(item_b, 2) ** 2)
            objective += (self.alpha / 2) * ((np.linalg.norm((H - U.transpose() @ C), 'fro')) ** 2) - (
                    self.beta / 2) * (
                             np.trace(H.transpose() @ B @ F)) + (self.eta / 2) * (np.linalg.norm(F - H, 'fro') ** 2)
            for user in self.trainset.all_users():
                items = [j for j, _ in self.trainset.ur[user]]
                for item, rating in self.trainset.ur[user]:
                    userImp = np.zeros(self.numFactors)
                    for tempitem in items:
                        userImp += Imp[:, tempitem]
                    userImp *= (1.0 / sqrt(len(items)))
                    predicted = self.trainset.global_mean + user_b[user] + item_b[item] + (userImp + U[:, user]).dot(
                        V[:, item])
                    error = predicted - rating
                    objective += error ** 2
                    dJbu = error + self.gamma * user_b[user]
                    user_b[user] -= self.learnrate * dJbu
                    dJbi = error + self.gamma * item_b[item]
                    item_b[item] -= self.learnrate * dJbi
                    dJu = error * V[:, item] + self.gamma * U[:, user]
                    U[:, user] -= self.learnrate * dJu
                    dJv = error * (U[:, user] + userImp) + self.gamma * V[:, item]
                    V[:, item] -= self.learnrate * dJv
                    for tempitem in items:
                        dJy = error * (1.0 / sqrt(len(items))) * V[:, item] + self.impitem * Imp[:, tempitem]
                        Imp[:, tempitem] -= self.learnrate * dJy
                U[:, user] -= self.learnrate * self.alpha * (CCT @ U[:, user] - CHT[:, user])
            # dJC = -self.alpha * U @ H + self.alpha * U @ U.transpose() @ C
            # C -= self.learnrate * dJC
            C = np.linalg.inv(U @ U.transpose()) @ U @ H
            Q = (self.alpha / (self.eta + self.alpha)) * C.transpose() @ U + (
                    self.beta / (2 * (self.eta + self.alpha))) * F.transpose() @ B.transpose() + (
                        self.eta / (self.eta + self.alpha)) * F.transpose()
            UH, DH, VHT = np.linalg.svd(Q, full_matrices=True)
            H = VHT.transpose() @ Inc @ UH.transpose()
            F = np.maximum(H + (self.beta / (2 * self.eta)) * B.transpose() @ H, 0)
            objective /= 2
            print("This is " + str(step + 1) + "th iteration, loss = " + str(objective) + ", delta loss = " + str(
                last_loss - objective))
            if fabs(last_loss - objective) <= 0.000001:
                print("Converged!!!")
                break
            last_loss = objective
            # self.test(U, V, user_b, item_b, Imp)
        return U, V, user_b, item_b, Imp, H

    def estimate(self, U, V, user_b, item_b, Imp, inn_uid, inn_iid):
        userImp = np.zeros(self.numFactors)
        items = [j for j, _ in self.trainset.ur[inn_uid]]
        for tempitem in items:
            userImp += Imp[:, tempitem]
        userImp *= (1.0 / sqrt(len(items)))
        predicted = self.trainset.global_mean + user_b[inn_uid] + item_b[inn_iid] + (userImp + U[:, inn_uid]).dot(
            V[:, inn_iid])
        predicted = min(self.rating_scale[1], predicted)
        predicted = max(self.rating_scale[0], predicted)
        return predicted

    def test(self, U, V, user_b, item_b, Imp):
        # Constructing predictions
        predictions = []
        for raw_uid, raw_iid, true_r in self.testset:
            try:
                if true_r != 0:
                    inn_uid = self.trainset.to_inner_uid(raw_uid)
                    inn_iid = self.trainset.to_inner_iid(raw_iid)
                    predicted = self.estimate(U, V, user_b, item_b, Imp, inn_uid, inn_iid)
                    # inner-uid, inner-iid, ground-truth, estimation, details
                    predictions.append(Prediction(inn_uid, inn_iid, true_r, predicted, {'was_impossible': False}))
            except ValueError:
                pass
        print(len(predictions), len(self.testset))
        MAE = getMAE(predictions)
        RMSE = getRMSE(predictions)
        print('MAE= ' + str(MAE) + ', RMSE= ' + str(RMSE))
        cMAE = getColdMAE(predictions, self.trainset.ur)
        cRMSE = getColdRMSE(predictions, self.trainset.ur)
        p, r, f = precision_recall_at_k(predictions, threshold=self.threshold)
        fpr, tpr = getROC(predictions, threshold=self.threshold)
        myauc = getAUC(fpr, tpr)
        ndcg = getNDCG(predictions, threshold=self.threshold)
        return MAE, RMSE, cMAE, cRMSE, p, r, f, fpr, tpr, myauc, ndcg, predictions


if __name__ == '__main__':
    tMAE = tRMSE = tcMAE = tcRMSE = 0
    tPre = tRec = tF1 = 0
    fprList = []
    tprList = []
    tAUC = tNDCG = 0
    datname = input('please enter dataset name(one of filmtrust, ciao, epinions2 and flixster): ')
    rmin = input('please enter the lower bound of rating scale: ')
    rmax = input('please enter the upper bound of rating scale: ')
    if input('Do you want to change parameters? ') == 'yes':
        numFactors = int(input("please enter the number of factors "))
        numCommunities = int(input('please enter the number of communities '))
        alpha = float(input('please enter the alpha '))
        beta = float(input('please enter the beta '))
        eta = float(input('please enter the eta '))
        gamma = float(input('please enter the gamma '))
        impitem = float(input('please enter the implicit item factor '))
        learnrate = float(input('please enter the learning rate '))
        maxiteration = int(input('please enter the maximum number of iteration '))
        thre = float(input('please enter the threshold for positive '))
        for i in range(5):
            print("Current fold is " + str(i + 1))
            rpath = "./Datasets/" + datname + "/cvtrain" + str(i + 1) + ".txt"
            vpath = "./Datasets/" + datname + "/cvtest" + str(i + 1) + ".txt"
            tpath = "./Datasets/" + datname + "/trust.txt"
            obj = SCMF(rpath, vpath, tpath, rmin, rmax)
            obj.numFactors = numFactors
            obj.numCommunities = numCommunities
            obj.alpha = alpha
            obj.beta = beta
            obj.eta = eta
            obj.gamma = gamma
            obj.impitem = impitem
            obj.learnrate = learnrate
            obj.maxiteration = maxiteration
            obj.threshold = thre
            U, V, user_b, item_b, Imp, H = obj.train()
            MAE, RMSE, cMAE, cRMSE, p, r, f, fpr, tpr, myauc, ndcg, mypreds = obj.test(U, V, user_b, item_b, Imp)
            tMAE += MAE
            tRMSE += RMSE
            tcMAE += cMAE
            tcRMSE += cRMSE
            tPre += p
            tRec += r
            tF1 += f
            fprList.append(fpr)
            tprList.append(tpr)
            tAUC += myauc
            tNDCG += ndcg
            np.save('./SCSVD_' + datname + '_' + str(numFactors) + '_' + str(i + 1) + '.npy', mypreds)
            np.save('./U_' + datname + str(i + 1) + '_' + str(numFactors) + '.npy', U)
            np.save('./H_' + datname + str(i + 1) + '_' + str(numFactors) + '.npy', H)
            obj.print_parameters()
        print("Fianlly, MAE = " + str(tMAE / 5) + ", RMSE = " + str(tRMSE / 5))
        print("cMAE = " + str(tcMAE / 5) + ", cRMSE = " + str(tcRMSE / 5))
        print("pre = " + str(tPre / 5) + ", rec = " + str(tRec / 5) + ", f1 = " + str(tF1 / 5))
        print("AUC = " + str(tAUC / 5) + ", NDCG = " + str(tNDCG / 5))
        np.save('./SCSVD_tpr_' + datname + '_ave_' + str(numFactors) + '.npy', tprList)
        np.save('./SCSVD_fpr_' + datname + '_ave_' + str(numFactors) + '.npy', fprList)
        print(datname)
    else:
        for i in range(5):
            print("Current fold is " + str(i + 1))
            rpath = "./Datasets/" + datname + "/cvtrain" + str(i + 1) + ".txt"
            vpath = "./Datasets/" + datname + "/cvtest" + str(i + 1) + ".txt"
            tpath = "./Datasets/" + datname + "/trust.txt"
            obj = SCMF(rpath, vpath, tpath, rmin, rmax)
            U, V, user_b, item_b, Imp, H = obj.train()
            MAE, RMSE, cMAE, cRMSE, p, r, f, fpr, tpr, myauc, ndcg, mypreds = obj.test(U, V, user_b, item_b, Imp)
            tMAE += MAE
            tRMSE += RMSE
            tcMAE += cMAE
            tcRMSE += cRMSE
            tPre += p
            tRec += r
            tF1 += f
            fprList.append(fpr)
            tprList.append(tpr)
            tAUC += myauc
            tNDCG += ndcg
            np.save('./SCSVD_' + datname + '_' + str(10) + '_' + str(i + 1) + '.npy', mypreds)
            np.save('./U_' + datname + str(i + 1) + '_' + str(10) + '.npy', U)
            np.save('./H_' + datname + str(i + 1) + '_' + str(10) + '.npy', H)
            obj.print_parameters()
        print("Fianlly, MAE = " + str(tMAE / 5) + ", RMSE = " + str(tRMSE / 5))
        print("cMAE = " + str(tcMAE / 5) + ", cRMSE = " + str(tcRMSE / 5))
        print("pre = " + str(tPre / 5) + ", rec = " + str(tRec / 5) + ", f1 = " + str(tF1 / 5))
        print("AUC = " + str(tAUC / 5) + ", NDCG = " + str(tNDCG / 5))
        np.save('./SCSVD_tpr_' + datname + '_ave_' + str(10) + '.npy', tprList)
        np.save('./SCSVD_fpr_' + datname + '_ave_' + str(10) + '.npy', fprList)
        print(datname)
    print('Ended!')
