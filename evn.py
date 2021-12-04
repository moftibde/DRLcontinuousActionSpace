import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patheffects as path_effects


class action_space(object):
    def __init__(self, torque_max):
        self.action_space = np.array([1])
        self.high = np.array([torque_max])
        self.shape = self.action_space.shape

    def sample(self):
        return (np.random.random_sample()-0.5)*self.high*2


class act_space(object):
    def __init__(self, epsmax, act):
        self.act = act
        self.high = epsmax
        if act == 'optimize':
            self.action_space = np.zeros((5))
            self.shape = self.action_space.shape
        elif act == 'self_control':
            self.action_space = np.zeros((1))
            self.shape = self.action_space.shape
        else:
            raise NotImplementedError

    def sample(self):
        if self.act == 'optimize':
            alpha1, alpha2, lambda1, lambda2, epsilon = np.random.random_sample(
                5)
            # epsilon = (np.random.rand()-0.5)*2*self.high
            return np.array([alpha1, alpha2, lambda1, lambda2, epsilon])
        elif self.act == 'self_control':
            fc = np.random.random_sample(1)*2*self.high-self.high
            return fc


class main_action_space():
    def __init__(self, fxm, fym, fzm, act):
        self.act = act
        self.highx = fxm 
        self.highy = fym 
        self.highz = fzm 
        if self.act == 'optimize':
            self.action_space = np.zeros((15))
            self.shape = self.action_space.shape

        elif self.act == 'self_control':
            self.action_space = np.zeros((3))
            self.shape = self.action_space.shape
        else:
            raise NotImplementedError
    
    def sample(self):
        if self.act == 'optimize':
            return np.random.rand(15)
        elif self.act == 'self_control':
            return np.random.rand(3) 


class pendenv(object):
    def __init__(self):
        self.reset()
        self.action_space = action_space()

    def step(self, torque):
        x = self.x
        i = self.i
        self.torque = torque
        if i >= 2:
            x1 = x[i-1, 0]
            dx1 = x[i-1, 1]
            dx2 = x[i-2, 1]
            ddx1 = x[i-1, 2]
            ddx2 = x[i-2, 2]
            x[i, 0] = x1+3*self.T/2*dx1-self.T/2*dx2
            x[i, 1] = dx1+3*self.T/2*ddx1-self.T/2*ddx2
            x[i, 2] = (self.torque - self.m*self.g*self.l*np.cos(x[i, 0]) -
                       self.k*x[i, 0]-self.c*x[i, 1])/self.JG
            x0 = x[i, 0]
            dx0 = x[i, 1]
            ddx0 = x[i, 2]
        else:
            x1 = x[i-1, 0]
            dx1 = x[i-1, 1]
            ddx1 = x[i-1, 2]
            x[i, 0] = x1+3*self.T/2*dx1
            x[i, 1] = dx1+3*self.T/2*ddx1
            x[i, 2] = (self.torque - self.m*self.g*self.l*np.cos(x[i, 0]) -
                       self.k*x[i, 0]-self.c*x[i, 1])/self.JG
            x0 = x[i, 0]
            dx0 = x[i, 1]
            ddx0 = x[i, 2]

        self.i += 1
        self.state_space = np.array([x0, dx0, self.xd[i], self.dxd[i]])
        self.reward = self.get_reward(
            x0, dx0, torque=self.torque, type='speed and position')
        self.done = True if i == self.Nc - 1 else False
        return self.state_space, self.reward, self.done, self.blank

    def get_reward(self, x0, dx0, torque, type='position'):
        # sometimes it is necessary to change the reward funciton
        # i will come to this
        if type == 'position':
            # + np.min([(.1/abs(x0 - self.xd[self.i - 1]))**0.1, 1000])
            reward = -(abs(x0 - self.xd[self.i - 1]))**1.5
            reward -= 0.0001*abs(torque)
            self.reward_list[self.i - 1] = reward
        elif type == 'speed and position':
            reward = - \
                (10*abs(x0 - self.xd[self.i - 1]) +
                 0.1*abs(dx0 - self.dxd[self.i - 1]))
            reward -= 0.01*abs(torque)
            self.reward_list[self.i - 1] = reward
        else:
            raise ValueError('please ensure to set a currect type of type :/')
        return reward

    def reset(self, k=3, c=1, m=2, g=9.806, l=0.5, tstop=3, T=0.01, torque_max=300, xdf=.6, xd0=2, destype='sine'):
        self.blank = None
        self.i = 1
        self.k = k
        self.c = c
        self.m = m
        self.g = g
        self.l = l
        self.JG = self.m*self.l**2
        self.tstop = tstop
        self.T = T
        self.destype = destype
        self.xd0 = xd0
        self.xdf = xdf
        self.Nc = int(np.ceil(self.tstop/self.T))
        self.t = np.linspace(0, self.tstop, self.Nc)
        self.reward_list = np.zeros_like(self.t)
        if self.destype == 'sine':
            self.xd = self.xd0*np.sin(2*np.pi*self.xdf*self.t)
        elif self.destype == 'sign':
            self.xd = np.sign(self.xd0*np.sin(2*np.pi*self.xdf*self.t))
        self.dxd = (np.diff(self.xd)/self.T)
        self.dxd = np.append(self.dxd, [0])
        self.torque_max = torque_max
        self.x = np.zeros((self.Nc, 3))
        # self.xd = desth0*np.sign(np.sin(2*np.pi*desthfreq*self.t)) + desth0*0.1*np.random.rand(self.Nc)
        # self.dxd = np.diff(self.xd)/self.T

        x0 = 10*np.pi/180
        dx0 = 0
        self.x[0, :] = [x0, dx0, 0]
        self.state_space = np.array(
            [self.x[self.i, 0], self.x[self.i, 1], self.xd[self.i], self.dxd[self.i]])
        return self.state_space


class mainmodel(object):
    def __init__(self):
        self.reset()
        self.action_space = main_action_space(self.UC1, self.UC2, self.UC3, self.act)

    def reseti(self, x, arg, first=False):
        if not first:
            x1 = x[arg-1, 1]
            y1 = x[arg-1, 4]
            th1 = x[arg-1, 7]
            x2 = x[arg-2, 1]
            y2 = x[arg-2, 4]
            th2 = x[arg-2, 7]

            dx1 = x[arg-1, 2]
            dy1 = x[arg-1, 5]
            dth1 = x[arg-1, 8]
            dx2 = x[arg-2, 2]
            dy2 = x[arg-2, 5]
            dth2 = x[arg-2, 8]

            ddx2 = x[arg-2, 3]
            ddy2 = x[arg-2, 6]
            ddth2 = x[arg-2, 9]

            ddx1 = x[arg-1, 3]
            ddy1 = x[arg-1, 6]
            ddth1 = x[arg-1, 9]
            return x1, y1, th1, x2, y2, th2, dx1, dy1, dth1, dx2, dy2, dth2, ddx1, ddy1, ddth1, ddx2, ddy2, ddth2
        else:
            x1 = x[arg-1, 1]
            y1 = x[arg-1, 4]
            th1 = x[arg-1, 7]

            dx1 = x[arg-1, 2]
            dy1 = x[arg-1, 5]
            dth1 = x[arg-1, 8]

            ddx1 = x[arg-1, 3]
            ddy1 = x[arg-1, 6]
            ddth1 = x[arg-1, 9]
            return x1, y1, th1, dx1, dy1, dth1, ddx1, ddy1, ddth1

    def setM(self, th):
        MC, ML, mb, JG, L = self.MC, self.ML, self.mb, self.JG, self.L
        m11 = MC + ML + 2*mb
        m12 = 0.0
        m13 = -(ML + mb)*L*np.sin(th)
        m21 = 0.0
        m22 = MC + ML + 2 * mb
        m23 = (ML + mb)*L*np.cos(th)
        m31 = -(ML + mb)*L*np.sin(th)
        m32 = (ML + mb)*L*np.cos(th)
        m33 = ML*(L**2) + mb*(L**2)/2 + 2*JG
        mMat = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
        return mMat

    def setC(self, th, dth):
        MC, ML, mb, L, g = self.MC, self.ML, self.mb, self.L, self.g
        c1 = (ML + mb)*L*dth**2*np.cos(th)
        c2 = -(MC + ML + 2*mb)*g + (ML + mb)*L*dth**2*np.sin(th)
        c3 = -(ML + mb)*g*L*np.cos(th)
        cVec = np.array([[c1, c2, c3]]).transpose()
        return cVec

    def makei(self, A, ql, qh):
        # print(A,ql,qh)
        A, ql, qh = self.sortmatrices(A, ql, qh)
        A[1, :] -= (A[1, 0]/A[0, 0])*A[0, :]
        ql[1] -= (A[1, 0]/A[0, 0])*ql[0]
        qh[1] -= (A[1, 0]/A[0, 0])*qh[0]
        A, ql, qh = self.sortmatrices(A, ql, qh)
        A[2, :] -= (A[2, 0]/A[0, 0])*A[0, :]
        ql[2] -= (A[2, 0]/A[0, 0])*ql[0]
        qh[2] -= (A[2, 0]/A[0, 0])*qh[0]
        A, ql, qh = self.sortmatrices(A, ql, qh)
        A[2, :] -= (A[2, 1]/A[1, 1])*A[1, :]
        ql[2] -= (A[2, 1]/A[1, 1])*ql[1]
        qh[2] -= (A[2, 1]/A[1, 1])*qh[1]
        A, ql, qh = self.sortmatrices(A, ql, qh)
        A[1, :] -= (A[1, 2]/A[2, 2])*A[2, :]
        ql[1] -= (A[1, 2]/A[2, 2])*ql[2]
        qh[1] -= (A[1, 2]/A[2, 2])*qh[2]
        A, ql, qh = self.sortmatrices(A, ql, qh)
        A[0, :] -= (A[0, 1]/A[1, 1])*A[1, :]
        ql[0] -= (A[0, 1]/A[1, 1])*ql[1]
        qh[0] -= (A[0, 1]/A[1, 1])*qh[1]
        A, ql, qh = self.sortmatrices(A, ql, qh)
        A[0, :] -= (A[0, 2]/A[2, 2])*A[2, :]
        ql[0] -= (A[0, 2]/A[2, 2])*ql[2]
        qh[0] -= (A[0, 2]/A[2, 2])*qh[2]
        # print('processing\n',A)
        A, ql, qh = self.normali(A, ql, qh)
    #     print('the matrices are:','\n',A,'\n',ql,'\n',qh)
        return ql, qh, A

    def switchq(self, ql, qh, arr):
        a = ql[arr, 0]
        ql[arr, 0] = qh[arr, 0]
        qh[arr, 0] = a
        return (ql, qh)

    def sortmatrices(self, A, ql, qh):
        j = 0
        for i in np.diag(A):
            if i == 0:
                counter = 0
                for arr in A[:, j]:
                    if arr != 0 and A[j, counter] != 0:
                        A[[counter, j]] = A[[j, counter]]
                        ql[[counter, j]] = ql[[j, counter]]
                        qh[[counter, j]] = qh[[j, counter]]
                        # print('row ', j, counter,' successfully switched','\n',ql,'\n',qh,'\n',A)
                        break
                    counter += 1
            j += 1
        return A, ql, qh

    def normali(self, A, ql, qh):
        for i in range(A.shape[0]):
            ql[i, 0] = ql[i, 0] / abs(A[i, i])
            qh[i, 0] = qh[i, 0] / abs(A[i, i])
            sign = np.sign(A[i, i])
            if sign >= 0:
                pass
            else:
                self.switchq(ql, qh, i)
            A[i, i] /= A[i, i]
        # print('normalizing\n',A)
        return A, ql, qh

    def requirement(self, act):
        if act == 'optimize':

            return self.i, self.x, self.T, self.xd, self.yd, self.thd, self.dxd, self.dyd, self.dthd, self.BI, self.QA, self.QB, self.qlfpd, self.qhfpd, self.qlfvd, self.qhfvd, self.sigmap1, self.sigmap2, self.sigmaa1, self.sigmaa2, self.UC1, self.UC2, self.UC3
        elif act == 'self_control':
            return self.i, self.x, self.T, self.xd, self.yd, self.thd, self.dxd, self.dyd, self.dthd, self.sigmap1, self.sigmap2, self.sigmaa1, self.sigmaa2, self.UC1, self.UC2, self.UC3

    def step(self, output):
        if self.act == 'optimize':
            i, x, T, xd, yd, thd, dxd, dyd, dthd, BI, QA, QB, qlfpd, qhfpd, qlfvd, qhfvd, sigmap1, sigmap2, sigmaa1, sigmaa2, UC1, UC2, UC3 = self.requirement(
                self.act)
            lambdaxp, lambdaxv, lambdayp, lambdayv, lambdathp, lambdathv, alphaxp, alphaxv, alphayp, alphayv, alphathp, alphathv, epsilonx, epsilony, epsilonth = output
            if not self.epst:
                epsilonx = True if epsilonx > 0.5 else False
                epsilony = True if epsilony > 0.5 else False
                epsilonth = True if epsilonth > 0.5 else False

        elif self.act == 'self_control':
            i, x, T, xd, yd, thd, dxd, dyd, dthd, sigmap1, sigmap2, sigmaa1, sigmaa2, UC1, UC2, UC3 = self.requirement(
                self.act)
            fcx1, fcy1, fcth1 = output
            fcx1 = fcx1*2*self.UC1 - self.UC1
            fcy1 = fcy1*2*self.UC2 - self.UC2
            fcth1 = fcth1*2*self.UC3 - self.UC3

        x[i, 0] = (i-1)*T
        if i >= 2:

            x1, y1, th1, x2, y2, th2, dx1, dy1, dth1, dx2, dy2, dth2, ddx1, ddy1, ddth1, ddx2, ddy2, ddth2 = self.reseti(
                x=x, arg=i)

            mMat1 = self.setM(th=th1)
            mMat2 = self.setM(th=th2)

            cVec1 = self.setC(th=th1, dth=dth1)
            cVec2 = self.setC(th=th2, dth=dth2)

            minv1 = np.linalg.inv(mMat1)
            G = np.matmul(minv1, cVec1)
            bi = np.matmul(minv1, np.eye(3))
            bi1 = bi.copy()
            bi2 = bi.copy()
            gx = G[0]
            gy = G[1]
            gth = G[2]

            if self.act == 'optimize':

                fcx2p = x[i-1, 13]
                fcy2p = x[i-1, 14]
                fcth2p = x[i-1, 15]
                fcx2v = x[i-1, 16]
                fcy2v = x[i-1, 17]
                fcth2v = x[i-1, 18]
                fx2p = x[i-1, 19]
                fy2p = x[i-1, 20]
                fth2p = x[i-1, 21]
                fx2v = x[i-1, 22]
                fy2v = x[i-1, 23]
                fth2v = x[i-1, 24]
                sxp = (1+lambdaxp)*x1+3*T/2*dx1-T/2*dx2-xd[i]-lambdaxp*xd[i-1]
                syp = (1+lambdayp)*y1+3*T/2*dy1-T/2*dy2-yd[i]-lambdayp*yd[i-1]
                sthp = (1+lambdathp)*x1+3*T/2*dth1-T / \
                    2*dth2-thd[i]-lambdathp*thd[i-1]
                sxv = dx1-dxd[i-1] + lambdaxv*(dx2-dxd[i-2])
                syv = dy1-dyd[i-1] + lambdayv*(dy2-dyd[i-2])
                sthv = dth1-dthd[i-1] + lambdathv*(dth2-dthd[i-2])

                kacixp = (1+lambdaxp)*(x1+3*T/2*dx1-T/2*dx2)+3*T/2 * \
                    (dx1+3*T/2*gx-T/2*ddx2)-T/2*dx1-xd[i+1]-lambdaxp*xd[i]
                kaciyp = (1+lambdayp)*(y1+3*T/2*dy1-T/2*dy2)+3*T/2 * \
                    (dy1+3*T/2*gy-T/2*ddy2)-T/2*dy1-yd[i+1]-lambdayp*yd[i]
                kacithp = (1+lambdathp)*(th1+3*T/2*dth1-T/2*dth2)+3*T/2 * \
                    (dth1+3*T/2*gth-T/2*ddth2)-T / \
                    2*dth1-thd[i+1]-lambdathp*thd[i]
                kacixv = (1+lambdaxv)*dx1+3*T/2*gx-T / \
                    2*ddx2-dxd[i]-lambdaxv*dxd[i-1]
                kaciyv = (1+lambdayv)*dy1+3*T/2*gy-T / \
                    2*ddy2-dyd[i]-lambdayv*dyd[i-1]
                kacithv = (1+lambdathv)*dth1+3*T/2*gth-T / \
                    2*ddth2-dthd[i]-lambdathv*dthd[i-1]

                qlxp = 4/(9*T**2)*(-kacixp-abs(sxp))
                qlyp = 4/(9*T**2)*(-kaciyp-abs(syp))
                qlthp = 4/(9*T**2)*(-kacithp-abs(sthp))
                qhxp = 4/(9*T**2)*(-kacixp+abs(sxp))
                qhyp = 4/(9*T**2)*(-kaciyp+abs(syp))
                qhthp = 4/(9*T**2)*(-kacithp+abs(sthp))
                qlxv = 2/(3*T)*(-kacixv-abs(sxv))
                qlyv = 2/(3*T)*(-kaciyv-abs(syv))
                qlthv = 2/(3*T)*(-kacithv-abs(sthv))
                qhxv = 2/(3*T)*(-kacixv+abs(sxv))
                qhyv = 2/(3*T)*(-kaciyv+abs(syv))
                qhthv = 2/(3*T)*(-kacithv+abs(sthv))

                qlp = np.array([qlxp, qlyp, qlthp])
                qhp = np.array([qhxp, qhyp, qhthp])
                qlv = np.array([qlxv, qlyv, qlthv])
                qhv = np.array([qhxv, qhyv, qhthv])
                BI[:, :, i] = bi
                QB[:, :, i] = np.hstack([qlp, qhp, qlv, qhv]).transpose()
                # if i <10:
                #     print(bi1)#, A, np.matmul(A, A),(np.matmul(A, A) == A).all())
                qlfp, qhfp, A = self.makei(bi1, ql=qlp, qh=qhp)
                # if i <10:
                #     print(bi1)
                # if i <10:
                #     print(bi2)
                qlfv, qhfv, A = self.makei(bi2, ql=qlv, qh=qhv)
                # if i <10:
                #     print(bi2)
                QA[:, :, i] = np.hstack([qlfp, qhfp, qlfv, qhfv]).transpose()

                qlfpd[0] = qlfp[0]/sigmap2-sigmap1*fx2p/sigmap2-fcx2p
                qlfpd[1] = qlfp[1]/sigmap2-sigmap1*fy2p/sigmap2-fcy2p
                qlfpd[2] = qlfp[2]/sigmaa2-sigmaa1*fth2p/sigmaa2-fcth2p
                qhfpd[0] = qhfp[0]/sigmap2-sigmap1*fx2p/sigmap2-fcx2p
                qhfpd[1] = qhfp[1]/sigmap2-sigmap1*fy2p/sigmap2-fcy2p
                qhfpd[2] = qhfp[2]/sigmaa2-sigmaa1*fth2p/sigmaa2-fcth2p
                qlfvd[0] = qlfv[0]/sigmap2-sigmap1*fx2v/sigmap2-fcx2v
                qlfvd[1] = qlfv[1]/sigmap2-sigmap1*fy2v/sigmap2-fcy2v
                qlfvd[2] = qlfv[2]/sigmaa2-sigmaa1*fth2v/sigmaa2-fcth2v
                qhfvd[0] = qhfv[0]/sigmap2-sigmap1*fx2v/sigmap2-fcx2v
                qhfvd[1] = qhfv[1]/sigmap2-sigmap1*fy2v/sigmap2-fcy2v
                qhfvd[2] = qhfv[2]/sigmaa2-sigmaa1*fth2v/sigmaa2-fcth2v

                fcx1p = (1-alphaxp)*qlfpd[0] + alphaxp*qhfpd[0]
                fcy1p = (1-alphayp)*qlfpd[1] + alphayp*qhfpd[1]
                fcth1p = (1-alphathp)*qlfpd[2] + alphathp*qhfpd[2]

                fcx1v = (1-alphaxv)*qlfvd[0] + alphaxv*qhfvd[0]
                fcy1v = (1-alphayv)*qlfvd[1] + alphayv*qhfvd[1]
                fcth1v = (1-alphathv)*qlfvd[2] + alphathv*qhfvd[2]

                if abs(fcx1p) > UC1:
                    fcx1p = UC1*np.sign(fcx1p)
                if abs(fcx1v) > UC1:
                    fcx1v = UC1*np.sign(fcx1v)

                if abs(fcy1p) > UC2:
                    fcy1p = UC2*np.sign(fcy1p)
                if abs(fcy1v) > UC2:
                    fcy1v = UC2*np.sign(fcy1v)

                if abs(fcth1p) > UC3:
                    fcth1p = UC3*np.sign(fcth1p)
                if abs(fcth1v) > UC3:
                    fcth1v = UC3*np.sign(fcth1v)
                # calculate the kinematic parameters
                fx1p = sigmap1*fx2p+sigmap2*fcx1p+sigmap2*fcx2p
                fy1p = sigmap1*fy2p+sigmap2*fcy1p+sigmap2*fcy2p
                fth1p = sigmaa1*fth2p+sigmaa2*fcth1p+sigmaa2*fcth2p

                fx1v = sigmap1*fx2v+sigmap2*fcx1v+sigmap2*fcx2v
                fy1v = sigmap1*fy2v+sigmap2*fcy1v+sigmap2*fcy2v
                fth1v = sigmaa1*fth2v+sigmaa2*fcth1v+sigmaa2*fcth2v

                if abs(sxp) > epsilonx:  # and i%3!=0 or i%7==0:
                    flagx = 1
                    fx1 = fx1p
                    fcx1 = fcx1p
                else:
                    flagx = 2
                    fx1 = fx1v
                    fcx1 = fcx1v

                if abs(syp) > epsilony:  # and i%3!=0 or i%7==0:
                    flagy = 1
                    fy1 = fy1p
                    fcy1 = fcy1p
                else:
                    flagy = 2
                    fy1 = fy1v
                    fcy1 = fcy1v

                if abs(sthp) > epsilonth:  # and i%3!=0 or i%7==0:
                    flagth = 1
                    fth1 = fth1p
                    fcth1 = fcth1p
                else:
                    flagth = 2
                    fth1 = fth1v
                    fcth1 = fcth1v

            elif self.act == 'self_control':

                fcx2 = x[i-1, 13]
                fcy2 = x[i-1, 14]
                fcth2 = x[i-1, 15]
                fx2 = x[i-1, 16]
                fy2 = x[i-1, 17]
                fth2 = x[i-1, 18]

                fx1 = sigmap1*fx2+sigmap2*fcx1+sigmap2*fcx2
                fy1 = sigmap1*fy2+sigmap2*fcy1+sigmap2*fcy2
                fth1 = sigmaa1*fth2+sigmaa2*fcth1+sigmaa2*fcth2

            else:
                raise NotImplementedError

            u1 = np.array([fx1, fy1, fth1])
            # dd = np.linalg.inv(mMat1)*(cVec1 + np.eye(3)*u1)
            # dd = np.matmul(minv1, (cVec1 + np.matmul(np.eye(3), u1)))
            dd = G.T+np.matmul(bi, u1)
            # print(dd)
            dd = dd[0]
            # print(dd)
            ddx0 = dd[0]
            ddy0 = dd[1]
            ddth0 = dd[2]

            x0 = x[i, 1] = x[i-1, 1] + 3*T/2*dx1 - T/2*dx2
            dx0 = x[i, 2] = x[i-1, 2] + 3*T/2*ddx1 - T/2*ddx2
            x[i, 3] = ddx0

            y0 = x[i, 4] = x[i-1, 4] + 3*T/2*dy1 - T/2*dy2
            dy0 = x[i, 5] = x[i-1, 5] + 3*T/2*ddy1 - T/2*ddy2
            x[i, 6] = ddy0

            th0 = x[i, 7] = x[i-1, 7] + 3*T/2*dth1 - T/2*dth2
            dth0 = x[i, 8] = x[i-1, 8] + 3*T/2*ddth1 - T/2*ddth2
            x[i, 9] = ddth0
            # if i % 100 == 0:
            #     print(i, sxp)

            if self.act == 'optimize':

                x[i, 10] = flagx
                x[i, 11] = flagy
                x[i, 12] = flagth
                x[i, 13] = fcx1p    # fcx
                x[i, 14] = fcy1p    # fcy
                x[i, 15] = fcth1p   # fcth
                x[i, 16] = fcx1v    # fcx
                x[i, 17] = fcy1v    # fcy
                x[i, 18] = fcth1v   # fcth
                x[i, 19] = fx1p     # fx
                x[i, 20] = fy1p     # fy
                x[i, 21] = fth1p    # fth
                x[i, 22] = fx1v     # fx
                x[i, 23] = fy1v     # fy
                x[i, 24] = fth1v    # fth
                x[i, 25] = epsilonx
                x[i, 26] = epsilony
                x[i, 27] = epsilonth
                x[i, 28] = sxp
                x[i, 29] = syp
                x[i, 30] = sthp
                x[i, 31] = sxv
                x[i, 32] = syv
                x[i, 33] = sthv
                x[i, 34] = fx1
                x[i, 35] = fy1
                x[i, 36] = fth1
                x[i, 37] = fcx1
                x[i, 38] = fcy1
                x[i, 39] = fcth1

            elif self.act == 'self_control':

                fcx2 = x[i-1, 10]
                fcy2 = x[i-1, 11]
                fcth2 = x[i-1, 12]
                fx2 = x[i-1, 13]
                fy2 = x[i-1, 14]
                fth2 = x[i-1, 15]
                x[i, 16] = fx1
                x[i, 17] = fy1
                x[i, 18] = fth1
                x[i, 19] = fcx1
                x[i, 20] = fcy1
                x[i, 21] = fcth1

        else:
            x1, y1, th1, dx1, dy1, dth1, ddx1, ddy1, ddth1 = self.reseti(
                x=x, arg=i, first=True)

            mMat1 = self.setM(th=th1)

            cVec1 = self.setC(th=th1, dth=dth1)
            minv1 = np.linalg.inv(mMat1)
            G = np.matmul(minv1, cVec1)
            bi = np.matmul(minv1, np.eye(3))

            gx = G[0]
            gy = G[1]
            gth = G[2]
            if self.act == 'optimize':
                fcx2p = x[i-1, 13]
                fcy2p = x[i-1, 14]
                fcth2p = x[i-1, 15]
                fcx2v = x[i-1, 16]
                fcy2v = x[i-1, 17]
                fcth2v = x[i-1, 18]
                fx2p = x[i-1, 19]
                fy2p = x[i-1, 20]
                fth2p = x[i-1, 21]
                fx2v = x[i-1, 22]
                fy2v = x[i-1, 23]
                fth2v = x[i-1, 24]

                sxp = (1+lambdaxp)*x1+3*T/2*dx1-xd[i]-lambdaxp*xd[i-1]
                syp = (1+lambdayp)*y1+3*T/2*dy1-yd[i]-lambdayp*yd[i-1]
                sthp = (1+lambdathp)*x1+3*T/2*dth1-thd[i]-lambdathp*thd[i-1]
                sxv = dx1-dxd[i-1]
                syv = dy1-dyd[i-1]
                sthv = dth1-dthd[i-1]

                kacixp = (1+lambdaxp)*(x1+3*T/2*dx1)+3*T/2 * \
                    (dx1+3*T/2*gx)-T/2*dx1-xd[i+1]-lambdaxp*xd[i]
                kaciyp = (1+lambdayp)*(y1+3*T/2*dy1)+3*T/2 * \
                    (dy1+3*T/2*gy)-T/2*dy1-yd[i+1]-lambdayp*yd[i]
                kacithp = (1+lambdathp)*(th1+3*T/2*dth1)+3*T/2 * \
                    (dth1+3*T/2*gth)-T/2*dth1-thd[i+1]-lambdathp*thd[i]
                kacixv = (1+lambdaxv)*dx1+3*T/2*gx-dxd[i]-lambdaxv*dxd[i-1]
                kaciyv = (1+lambdayv)*dy1+3*T/2*gy-dyd[i]-lambdayv*dyd[i-1]
                kacithv = (1+lambdathv)*dth1+3*T/2*gth-dthd[i]-lambdathv*dthd[i-1]

                qlxp = 4/(9*T**2)*(-kacixp-abs(sxp))
                qlyp = 4/(9*T**2)*(-kaciyp-abs(syp))
                qlthp = 4/(9*T**2)*(-kacithp-abs(sthp))
                qhxp = 4/(9*T**2)*(-kacixp+abs(sxp))
                qhyp = 4/(9*T**2)*(-kaciyp+abs(syp))
                qhthp = 4/(9*T**2)*(-kacithp+abs(sthp))
                qlxv = 2/(3*T)*(-kacixv-abs(sxv))
                qlyv = 2/(3*T)*(-kaciyv-abs(syv))
                qlthv = 2/(3*T)*(-kacithv-abs(sthv))
                qhxv = 2/(3*T)*(-kacixv+abs(sxv))
                qhyv = 2/(3*T)*(-kaciyv+abs(syv))
                qhthv = 2/(3*T)*(-kacithv+abs(sthv))

                qlp = np.array([qlxp, qlyp, qlthp])
                qhp = np.array([qhxp, qhyp, qhthp])
                qlv = np.array([qlxv, qlyv, qlthv])
                qhv = np.array([qhxv, qhyv, qhthv])

                # this code will be changed due to implementing the epsilon

                # =========================================================
                # conditions for implemengint the epsilon
                # =========================================================

                # now let's use constraint

                # decouple the matrix (important)

                qlfp, qhfp, A = self.makei(A=bi, ql=qlp, qh=qhp)
                qlfv, qhfv, A = self.makei(A=bi, ql=qlv, qh=qhv)

                qlfpd[0] = qlfp[0]/sigmap2-sigmap1*fx2p/sigmap2-fcx2p
                qlfpd[1] = qlfp[1]/sigmap2-sigmap1*fy2p/sigmap2-fcy2p
                qlfpd[2] = qlfp[2]/sigmaa2-sigmaa1*fth2p/sigmaa2-fcth2p
                qhfpd[0] = qhfp[0]/sigmap2-sigmap1*fx2p/sigmap2-fcx2p
                qhfpd[1] = qhfp[1]/sigmap2-sigmap1*fy2p/sigmap2-fcy2p
                qhfpd[2] = qhfp[2]/sigmaa2-sigmaa1*fth2p/sigmaa2-fcth2p
                qlfvd[0] = qlfv[0]/sigmap2-sigmap1*fx2v/sigmap2-fcx2v
                qlfvd[1] = qlfv[1]/sigmap2-sigmap1*fy2v/sigmap2-fcy2v
                qlfvd[2] = qlfv[2]/sigmaa2-sigmaa1*fth2v/sigmaa2-fcth2v
                qhfvd[0] = qhfv[0]/sigmap2-sigmap1*fx2v/sigmap2-fcx2v
                qhfvd[1] = qhfv[1]/sigmap2-sigmap1*fy2v/sigmap2-fcy2v
                qhfvd[2] = qhfv[2]/sigmaa2-sigmaa1*fth2v/sigmaa2-fcth2v

                fcx1p = (1-alphaxp)*qlfpd[0] + alphaxp*qhfpd[0]
                fcy1p = (1-alphayp)*qlfpd[1] + alphayp*qhfpd[1]
                fcth1p = (1-alphathp)*qlfpd[2] + alphathp*qhfpd[2]

                fcx1v = (1-alphaxv)*qlfvd[0] + alphaxv*qhfvd[0]
                fcy1v = (1-alphayv)*qlfvd[1] + alphayv*qhfvd[1]
                fcth1v = (1-alphathv)*qlfvd[2] + alphathv*qhfvd[2]

                fcx2p = x[i-1, 13]
                fcy2p = x[i-1, 14]
                fcth2p = x[i-1, 15]
                fcx2v = x[i-1, 16]
                fcy2v = x[i-1, 17]
                fcth2v = x[i-1, 18]
                fx2p = x[i-1, 19]
                fy2p = x[i-1, 20]
                fth2p = x[i-1, 21]
                fx2v = x[i-1, 22]
                fy2v = x[i-1, 23]
                fth2v = x[i-1, 24]

                if abs(fcx1p) > UC1:
                    fcx1p = UC1*np.sign(fcx1p)
                if abs(fcx1v) > UC1:
                    fcx1v = UC1*np.sign(fcx1v)

                if abs(fcy1p) > UC2:
                    fcy1p = UC2*np.sign(fcy1p)
                if abs(fcy1v) > UC2:
                    fcy1v = UC2*np.sign(fcy1v)

                if abs(fcth1p) > UC3:
                    fcth1p = UC3*np.sign(fcth1p)
                if abs(fcth1v) > UC3:
                    fcth1v = UC3*np.sign(fcth1v)
                # calculate the kinematic parameters

                fx1p = sigmap1*fx2p+sigmap2*fcx1p+sigmap2*fcx2p
                fy1p = sigmap1*fy2p+sigmap2*fcy1p+sigmap2*fcy2p
                fth1p = sigmaa1*fth2p+sigmaa2*fcth1p+sigmaa2*fcth2p

                fx1v = sigmap1*fx2v+sigmap2*fcx1v+sigmap2*fcx2v
                fy1v = sigmap1*fy2v+sigmap2*fcy1v+sigmap2*fcy2v
                fth1v = sigmaa1*fth2v+sigmaa2*fcth1v+sigmaa2*fcth2v

                if abs(sxp) > epsilonx:# and i%3!=0 or i%7==0:
                    flagx = 1
                    fx1 = fx1p
                    fcx1 = fcx1p
                else:
                    flagx = 2
                    fx1 = fx1v
                    fcx1 = fcx1v

                if abs(syp) > epsilony:# and i%3!=0 or i%7==0:
                    flagy = 1
                    fy1 = fy1p
                    fcy1 = fcy1p
                else:
                    flagy = 2
                    fy1 = fy1v
                    fcy1 = fcy1v

                if abs(sthp) > epsilonth:# and i%3!=0 or i%7==0:
                    flagth = 1
                    fth1 = fth1p
                    fcth1 = fcth1p
                else:
                    flagth = 2
                    fth1 = fth1v
                    fcth1 = fcth1v

            elif self.act == 'self_control':

                fcx2 = x[i-1, 13]
                fcy2 = x[i-1, 14]
                fcth2 = x[i-1, 15]
                fx2 = x[i-1, 16]
                fy2 = x[i-1, 17]
                fth2 = x[i-1, 18]

                fx1 = sigmap1*fx2+sigmap2*fcx1+sigmap2*fcx2
                fy1 = sigmap1*fy2+sigmap2*fcy1+sigmap2*fcy2
                fth1 = sigmaa1*fth2+sigmaa2*fcth1+sigmaa2*fcth2
            
            else:
                raise NotImplementedError

            u1 = np.array([fx1, fy1, fth1])
            # dd = np.linalg.inv(mMat1)*(cVec1 + np.eye(3)*u1)
            # dd = np.matmul(minv1, (cVec1 + np.matmul(np.eye(3), u1)))
            dd = G.T+np.matmul(bi, u1)
            print(dd)
            print(np.matmul(bi, u1))
            print(G.T)
            dd = dd[0]
            ddx0 = dd[0]
            ddy0 = dd[1]
            ddth0 = dd[2]

            x0 = x[i, 1] = x[i-1, 1] + 3*T/2*dx1
            dx0 = x[i, 2] = x[i-1, 2] + 3*T/2*ddx1
            x[i, 3] = ddx0

            y0 = x[i, 4] = x[i-1, 4] + 3*T/2*dy1
            dy0 = x[i, 5] = x[i-1, 5] + 3*T/2*ddy1
            x[i, 6] = ddy0

            th0 = x[i, 7] = x[i-1, 7] + 3*T/2*dth1
            dth0 = x[i, 8] = x[i-1, 8] + 3*T/2*ddth1
            x[i, 9] = ddth0
            if i % 100 == 0:
                print(i, sxp)
            if self.act == 'optimize':

                x[i, 10] = flagx
                x[i, 11] = flagy
                x[i, 12] = flagth
                x[i, 13] = fcx1p    # fcx
                x[i, 14] = fcy1p    # fcy
                x[i, 15] = fcth1p   # fcth
                x[i, 16] = fcx1v    # fcx
                x[i, 17] = fcy1v    # fcy
                x[i, 18] = fcth1v   # fcth
                x[i, 19] = fx1p     # fx
                x[i, 20] = fy1p     # fy
                x[i, 21] = fth1p    # fth
                x[i, 22] = fx1v     # fx
                x[i, 23] = fy1v     # fy
                x[i, 24] = fth1v    # fth
                x[i, 25] = epsilonx
                x[i, 26] = epsilony
                x[i, 27] = epsilonth
                x[i, 28] = sxp
                x[i, 29] = syp
                x[i, 30] = sthp
                x[i, 31] = sxv
                x[i, 32] = syv
                x[i, 33] = sthv
                x[i, 34] = fx1
                x[i, 35] = fy1
                x[i, 36] = fth1
                x[i, 37] = fcx1
                x[i, 38] = fcy1
                x[i, 39] = fcth1

            elif self.act == 'self_control':

                fcx2 = x[i-1, 10]
                fcy2 = x[i-1, 11]
                fcth2 = x[i-1, 12]
                fx2 = x[i-1, 13]
                fy2 = x[i-1, 14]
                fth2 = x[i-1, 15]
                x[i, 16] = fx1
                x[i, 17] = fy1
                x[i, 18] = fth1
                x[i, 19] = fcx1
                x[i, 20] = fcy1
                x[i, 21] = fcth1
        self.x = x
        self.i += 1
        if self.act == 'optimize':
            self.state_space = np.array(
                [x0, y0, th0, dx0, dy0, dth0, self.xd[self.i], self.yd[self.i], self.thd[self.i], self.dxd[self.i], self.dyd[self.i], self.dthd[self.i]])
        elif self.act == 'self_control':
            self.state_space = np.array(
                [x0, y0, th0, dx0, dy0, dth0, self.xd[self.i], self.yd[self.i], self.thd[self.i], self.dxd[self.i], self.dyd[self.i], self.dthd[self.i]])
        self.reward = self.get_reward(x, rtype='position')
        self.done = True if i == self.N - 1 else False
        return self.state_space, self.reward, self.done, self.blank

    def get_reward(self, x,rtype='position'):
        if rtype == 'position':
            reward = -(abs(4))

            self.reward_list[self.i - 1] = reward 
        elif rtype == 'speed and position':
            reward = - (abs(4))

        return  reward

    def reset(self, ML=3, MC=5, mb=1.5, L=1.0, l1=0.0, l2=0.0, g=0.0, tstop=1.3, T=0.005, UC1=1700, UC2=1500, UC3=1100, omegap=60, omegaa=60, Xdes0=1, Ydes0=1.5, Thdes0=30, freqX=1, freqY=0.7, freqTh=2, x0=0, y0=0, th0=0, dx0=0, dy0=0, dth0=0, destypex='sine', destypey='sine', destypeth='sine', act='optimize', epst=False):
        self.blank = None
        self.act = act
        self.epst = epst 
        self.i = 1
        self.ML = ML
        self.MC = MC
        self.mb = mb
        self.L = L
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.JG = 1/12*self.L**2*mb
        self.tstop = tstop
        self.T = T
        self.UC1 = UC1
        self.UC2 = UC2
        self.UC3 = UC3
        self.omegap = omegap*2*np.pi
        self.omegaa = omegaa*2*np.pi
        self.sigmap1 = (1-omegap*T/2)/(1+omegap*T/2)
        self.sigmap2 = (omegap*T/2)/(1+omegap*T/2)
        self.sigmaa1 = (1-omegaa*T/2)/(1+omegaa*T/2)
        self.sigmaa2 = (omegaa*T/2)/(1+omegaa*T/2)
        self.Xdes0 = Xdes0
        self.Ydes0 = Ydes0
        self.Thdes0 = Thdes0
        self.freqX = freqX
        self.freqY = freqY
        self.freqTh = freqTh
        self.x0 = x0
        self.y0 = y0
        self.th0 = th0
        self.dx0 = dx0
        self.dy0 = dy0
        self.dth0 = dth0
        self.N = int(np.ceil(self.tstop/self.T))
        self.t = np.linspace(0, self.tstop, self.N).transpose()
        self.reward_list = np.zeros((self.N))
        if destypex == 'sine':
            self.xd = self.Xdes0*np.sin(2*np.pi*freqX*self.t)
        elif destypex == 'sign':
            self.xd = self.Xdes0*np.sign(np.sin(2*np.pi*freqX*self.t))
        else:
            raise NotImplementedError
        self.xd = np.append(self.xd, [0])
        self.dxd = np.diff(self.xd)/self.T 
        self.dxd = np.append(self.dxd, [0])

        if destypey == 'sine':
            self.yd = self.Ydes0*np.sin(2*np.pi*freqY*self.t)
        elif destypey == 'sign':
            self.yd = self.Ydes0*np.sign(np.sin(2*np.pi*freqY*self.t))
        else:
            raise NotImplementedError
        self.yd = np.append(self.yd, [0])
        self.dyd = np.diff(self.yd)/self.T 
        self.dyd = np.append(self.dyd, [0])

        if destypeth == 'sine':
            self.thd = self.Thdes0*np.sin(2*np.pi*freqTh*self.t)
        elif destypeth == 'sign':
            self.thd = self.Thdes0*np.sign(np.sin(2*np.pi*freqTh*self.t))
        else:
            raise NotImplementedError
        self.thd = np.append(self.thd, [0])
        self.dthd = np.diff(self.thd)/self.T 
        self.dthd = np.append(self.dthd, [0])

        if self.act == 'optimize':
            self.state_space = np.array([self.x0, self.y0, self.th0, self.dx0, self.dy0, self.dth0, self.xd[self.i], self.yd[self.i], self.thd[self.i], self.dxd[self.i], self.dyd[self.i], self.dthd[self.i]])
            self.qlfpd = np.zeros((3, 1))
            self.qlfvd = np.zeros((3, 1))
            self.qhfpd = np.zeros((3, 1))
            self.qhfvd = np.zeros((3, 1))
            self.BI = np.zeros((3, 3, self.N))
            self.QB = np.zeros((4, 3, self.N))
            self.QA = np.zeros((4, 3, self.N))
            self.x = np.zeros((40, self.N)).transpose()
            self.x[0, :] = [0.0, self.x0, self.dx0, 0.0, self.y0, self.dy0, 0.0, self.th0, self.dth0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif self.act == 'self_control':
            self.state_space = np.array(
                [self.x0, self.y0, self.th0, self.dx0, self.dy0, self.dth0, self.xd[self.i], self.yd[self.i], self.thd[self.i], self.dxd[self.i], self.dyd[self.i], self.dthd[self.i]])
            self.x = np.zeros((22, self.N)).transpose()
            self.x[0, :] = [0.0, self.x0, self.dx0, 0.0, self.y0, self.dy0, 0.0, self.th0, self.dth0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        else:
            raise NotImplementedError

        return self.state_space


class simple1deps(object):
    def __init__(self):
        self.reset()
        self.action_space = act_space(self.epsmax, self.act)

    def requirement(self):
        return self.i, self.xol, self.xd, self.dxd, self.T, self.m, self.mu0, self.mu1, self.mu2, self.Fc, self.sigma1, self.sigma2, self.gamma1, self.gamma2

    def step(self, output):
        if self.act == 'optimize':
            alpha1, alpha2, lambda1, lambda2, epsilon = output
            if not self.epst:
                epsilon = True if epsilon > 0.5 else False
        elif self.act == 'self_control':
            fc1 = output*2*self.epsmax-self.epsmax
        else:
            raise NotImplementedError

        i, xol, xd, dxd, T, m, mu0, mu1, mu2, Fc, sigma1, sigma2, gamma1, gamma2 = self.requirement()
        if i >= 2:
            x1 = xol[i-1, 1]
            dx1 = xol[i-1, 2]
            dx2 = xol[i-2, 2]
            f2 = xol[i-2, 8]
            fc2 = xol[i-1, 9]
            g1 = -mu0/m*np.sign(dx1)-mu1/m*(dx1)-mu2/m*(dx1**2)*np.sign(dx1)
            g2 = -mu0/m*np.sign(dx2)-mu1/m*(dx2)-mu2/m*(dx2**2)*np.sign(dx2)
            if self.act == 'optimize':
                s1 = x1+3*T/2*dx1-T/2*dx2-xd[i]+lambda1*(x1-xd[i-1])
                s2 = dx1-dxd[i-1]+lambda2*(dx2-dxd[i-2])
                kaci1 = (1+lambda1)*(x1+3*T/2*dx1-T/2*dx2)+3*T/2*(dx1+3 *
                                                                  T/2*g1-T/2*(g2+1/m*f2))-T/2*dx1-xd[i+1]-lambda1*xd[i]
                kaci2 = dx1+3*T/2*g1-T/2 * \
                    (g2+1/m*f2)-dxd[i]+lambda2*(dx1-dxd[i-1])

                if self.epst:
                    if abs(s2) < epsilon:
                        flag = 1
                        QL1 = (-abs(s1)-kaci1)/gamma1
                        QH1 = (+abs(s1)-kaci1)/gamma1
                        fc1 = (1-alpha1)*QL1+alpha1*QH1
                    else:
                        flag = 2
                        QL2 = (-abs(s2)-kaci2)/gamma2
                        QH2 = (+abs(s2)-kaci2)/gamma2
                        fc1 = (1-alpha2)*QL2+alpha2*QH2
                else:
                    if epsilon:
                        flag = 1
                        QL1 = (-abs(s1)-kaci1)/gamma1
                        QH1 = (+abs(s1)-kaci1)/gamma1
                        fc1 = (1-alpha1)*QL1+alpha1*QH1
                    else:
                        flag = 2
                        QL2 = (-abs(s2)-kaci2)/gamma2
                        QH2 = (+abs(s2)-kaci2)/gamma2
                        fc1 = (1-alpha2)*QL2+alpha2*QH2

                if abs(fc1) >= Fc:
                    fc1 = Fc*np.sign(fc1)
                xol[i, 5] = epsilon
                xol[i, 6] = s2
                xol[i, 7] = alpha2
                xol[i-1, 10] = flag
            f1 = sigma1*f2 + sigma2*fc1 + sigma2*fc2
            ddx1 = g1 + 1/m*f1
            ddx2 = g2 + 1/m*f2

            x0 = xol[i, 1] = x1+3*T/2*dx1-T/2*dx2
            dx0 = xol[i, 2] = dx1+3*T/2*ddx1-T/2*ddx2
            xol[i, 3] = ddx1
            xol[i, 4] = g1
            xol[i-1, 8] = f1
            xol[i-1, 9] = fc1

        else:
            x1 = xol[i-1, 1]
            dx1 = xol[i-1, 2]
            fc2 = 0
            f2 = 0
            g1 = -mu0/m*np.sign(dx1)-mu1/m*(dx1)-mu2/m * \
                (dx1**2)*np.sign(dx1)
            if self.act == 'optimize':
                s1 = x1+3*T/2*dx1-xd[i]+lambda1*(x1-xd[i-1])
                s2 = dx1-dxd[i-1]
                kaci1 = (1+lambda1)*(x1+3*T/2*dx1)+3*T/2*(dx1+3*T/2 *
                                                          g1-T/2*(1/m*f2))-T/2*dx1-xd[i+1]-lambda1*xd[i]
                kaci2 = dx1+3*T/2*g1-T/2*(1/m*f2)-dxd[i]+lambda2*(dx1-dxd[i-1])
                if self.epst:
                    if abs(s2) < epsilon:
                        flag = 1
                        QL1 = (-abs(s1)-kaci1)/gamma1
                        QH1 = (+abs(s1)-kaci1)/gamma1
                        fc1 = (1-alpha1)*QL1+alpha1*QH1
                    else:
                        flag = 2
                        QL2 = (-abs(s2)-kaci2)/gamma2
                        QH2 = (+abs(s2)-kaci2)/gamma2
                        fc1 = (1-alpha2)*QL2+alpha2*QH2
                else:
                    if epsilon:
                        flag = 1
                        QL1 = (-abs(s1)-kaci1)/gamma1
                        QH1 = (+abs(s1)-kaci1)/gamma1
                        fc1 = (1-alpha1)*QL1+alpha1*QH1
                    else:
                        flag = 2
                        QL2 = (-abs(s2)-kaci2)/gamma2
                        QH2 = (+abs(s2)-kaci2)/gamma2
                        fc1 = (1-alpha2)*QL2+alpha2*QH2
                xol[i, 5] = epsilon
                xol[i, 6] = s2
                xol[i, 7] = alpha2
                xol[i-1, 10] = flag
            if abs(fc1) >= Fc:
                fc1 = Fc*np.sign(fc1)
            f1 = sigma1*f2 + sigma2*fc1 + sigma2*fc2
            ddx1 = g1 + 1/m*f1

            x0 = xol[i, 1] = x1+3*T/2*dx1
            dx0 = xol[i, 2] = dx1+3*T/2*ddx1
            xol[i, 3] = ddx1
            xol[i, 4] = g1
            xol[i-1, 8] = f1
            xol[i-1, 9] = fc1
        self.xol = xol
        self.i += 1
        if self.act == 'optimize':
            self.state_space = np.array(
                [x0, dx0, self.xd[i+1], self.dxd[i+1], f1, fc1, flag], dtype=np.float64)
        elif self.act == 'self_control':
            self.state_space = np.array(
                [x0, dx0, self.xd[i+1], self.dxd[i+1], f1, fc1], dtype=np.float32)
        self.reward = self.get_reward(
            x0, dx0, fc1, type='speed and position')
        self.done = True if i == self.Nc - 1 else False
        return self.state_space, self.reward, self.done, self.blank

    def get_reward(self, x0, dx0, Fc,  type='position'):
        if type == 'position':
            reward = -(abs(x0 - self.xd[self.i - 1]))**1.5
            reward -= 0.0001*abs(Fc)
            self.reward_list[self.i - 1] = reward
        elif type == 'speed and position':
            reward = - (abs(x0 - self.xd[self.i - 1]))
            #  *(max((1.4*(self.Nc - 2*self.i )/self.Nc)**10,0)+0.5)\
            #  + abs(dx0 - self.dxd[self.i - 1]))
            # reward -= .8*abs(Fc)
            # reward = -abs(Fc)
            self.reward_list[self.i - 1] = reward
        else:
            raise ValueError('please ensure to set a currect type of type :/')
        return reward

    def reset(self, m=1.5, mu0=10, mu1=5, mu2=3, tstop=1.3, T=0.001, freqBW=30, Fc=500, xd0=1, xdf=1, x0=0, dx0=0, epsmax=300, destype='sine', epst=False, act='self_control'):
        self.epst = epst
        self.blank = None
        self.act = act  # optimize or self_control
        self.i = 1
        self.m = m
        self.mu0 = mu0
        self.mu1 = mu1
        self.mu2 = mu2
        self.tstop = tstop
        self.T = T
        self.Nc = int(np.ceil(self.tstop/self.T))
        self.t = np.linspace(0, self.tstop, self.Nc)
        self.reward_list = np.zeros_like(self.t)
        self.freqBW = freqBW
        self.w = 2*np.pi*self.freqBW
        self.Fc = Fc
        self.destype = destype
        self.X0des = xd0
        self.xdesfreq = xdf
        if self.destype == 'sine':
            self.xd = self.X0des*np.sin(2*np.pi*self.xdesfreq*self.t)
        elif self.destype == 'sign':
            self.xd = np.sign(self.X0des*np.sin(2*np.pi*self.xdesfreq*self.t))
        self.xd = np.append(self.xd, [0])
        self.dxd = np.diff(self.xd)/self.T
        self.dxd = np.append(self.dxd, [0])
        self.sigma1 = (1-self.w*self.T/2)/(1+self.w*self.T/2)
        self.sigma2 = (self.w*self.T/2)/(1+self.w*self.T/2)

        self.x0 = x0
        self.dx0 = dx0

        self.xol = np.zeros((self.Nc, 12))
        self.xol[0, :] = np.array(
            [0, self.x0, self.dx0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.xol[:, 0] = self.t
        self.gamma1 = 9*self.T*self.T/4/self.m*self.sigma2
        self.gamma2 = 3*self.T/2/self.m*self.sigma2
        self.epsmax = epsmax
        # self.alpha1 = 0.55
        # self.alpha2 = 0.6
        # self.lambda1 = 0.8
        # self.lambda2 = 0.3
        # self.epsilon = 2
        if self.act == 'optimize':
            self.state_space = np.array(
                [x0, dx0, self.xd[self.i], self.dxd[self.i], self.Fc, self.Fc, 1])
        elif self.act == 'self_control':
            self.state_space = np.array(
                [x0, dx0, self.xd[self.i], self.dxd[self.i], self.Fc, self.Fc])
        return self.state_space


# test it nex
if __name__ == '__main__':
    # env = simple1deps()
    env = mainmodel()

    env.reset(tstop=3.3)
    a = env.action_space
    maxn = env.N
    s = np.zeros((maxn, env.state_space.shape[0]))
    r = np.zeros((maxn))
    d = np.zeros((maxn))
    # xd = env.xd
    torque = []
    print(a)
    print(maxn)
    for i in range(maxn-1):
        action = env.action_space.sample()
        torque.append(action)
        s[i, :], r[i], d[i], _ = env.step(action)
        print(d[i])

    plt.plot(s[:, 0], label='x')
    plt.plot(s[:, 1], label='y')
    plt.plot(s[:, 2], label='th')
    plt.plot(s[:, 3], label='dx')
    plt.plot(s[:, 4], label='dy')
    plt.plot(s[:, 5], label='dth')
    plt.plot(s[:, 6], label='xd')
    plt.plot(s[:, 7], label='yd')
    plt.plot(s[:, 8], label='thd')
    plt.plot(s[:, 9], label='dxd')
    plt.plot(s[:, 10], label='dyd')
    plt.plot(s[:, 11], label='dthd')
    plt.legend()
    plt.figure()
    plt.plot(s[:, 1], label='v')
    plt.plot(s[:, 3], label='dxd')
    plt.plot(s[:, 4], label='dxd')
    plt.plot(s[:, 5], label='dxd')
    # plt.plot(s[:, 6], label='dxd')
    # plt.plot(env.xol[:, 9], label='fc')
    plt.figure()
    plt.plot(r, label='reward')
    plt.legend()
    plt.figure()
    plt.plot(torque)
    plt.legend()
    plt.show()
    for i in range(30):
        b = a.sample()
        print(b)
